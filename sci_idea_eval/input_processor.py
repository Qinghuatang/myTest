"""
Input processor module for Sci-Idea-EVAL system.
Handles MCT extraction from ideas using Ollama LLM and vectorization.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from .config import Config, OllamaConfig
from .embedding_model import EmbeddingModel
from .utils import get_parent_class

logger = logging.getLogger(__name__)


@dataclass
class MCTTriple:
    """Represents extracted Mechanism-Context-Task triple."""
    mechanisms: List[str]
    contexts: List[str]
    tasks: List[str]
    mechanism_embeddings: Optional[Dict[str, np.ndarray]] = None
    context_embeddings: Optional[Dict[str, np.ndarray]] = None
    task_embeddings: Optional[Dict[str, np.ndarray]] = None
    parent_classes: Optional[Dict[str, str]] = None
    parent_embeddings: Optional[Dict[str, np.ndarray]] = None


class InputProcessor:
    """
    Processor for extracting MCT triples from idea text and converting to vectors.
    
    Uses Ollama with llama3.1:70b for MCT extraction and SciBERT/Specter for embeddings.
    """
    
    # Prompt template for MCT extraction
    MCT_EXTRACTION_PROMPT = """You are an expert in analyzing scientific research ideas. 
Your task is to extract the Mechanism-Context-Task (MCT) components from a research idea.

Definitions:
- Mechanism (M): The core method, technique, algorithm, or approach proposed in the idea.
- Context (C): The domain, setting, constraints, or conditions under which the idea operates.
- Task (T): The specific problem, goal, or objective the idea aims to solve.

Please analyze the following research idea and extract its MCT components.
Return your answer in the following JSON format:
```json
{
    "mechanism": ["list of mechanisms"],
    "context": ["list of contexts"],
    "task": ["list of tasks"]
}
```

Research Idea:
{idea_text}

Extract the MCT components:"""
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the input processor.
        
        Args:
            config: Configuration object.
            embedding_model: Pre-initialized embedding model.
        """
        self.config = config or Config()
        self.embedding_model = embedding_model or EmbeddingModel(self.config.embedding)
        
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate response.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            Generated response text.
        """
        url = f"{self.config.ollama.base_url}/api/generate"
        
        payload = {
            "model": self.config.ollama.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            }
        }
        
        for attempt in range(self.config.ollama.max_retries):
            try:
                response = requests.post(
                    url, 
                    json=payload,
                    timeout=self.config.ollama.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
            except requests.RequestException as e:
                logger.warning(f"Ollama API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.ollama.max_retries - 1:
                    raise
        
        return ""
    
    def _parse_mct_response(self, response: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Parse MCT extraction response from LLM.
        
        Args:
            response: LLM response text.
            
        Returns:
            Tuple of (mechanisms, contexts, tasks) lists.
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*"mechanism"[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group()
                # Handle potential JSON issues
                json_str = json_str.replace("'", '"')
                data = json.loads(json_str)
                
                mechanisms = data.get("mechanism", [])
                contexts = data.get("context", [])
                tasks = data.get("task", [])
                
                # Ensure lists
                if isinstance(mechanisms, str):
                    mechanisms = [mechanisms]
                if isinstance(contexts, str):
                    contexts = [contexts]
                if isinstance(tasks, str):
                    tasks = [tasks]
                
                return mechanisms, contexts, tasks
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract from structured text
        mechanisms = []
        contexts = []
        tasks = []
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            lower_line = line.lower()
            
            if "mechanism" in lower_line:
                current_section = "mechanism"
            elif "context" in lower_line:
                current_section = "context"
            elif "task" in lower_line:
                current_section = "task"
            elif line.startswith("-") or line.startswith("*"):
                item = line.lstrip("-*").strip()
                if item:
                    if current_section == "mechanism":
                        mechanisms.append(item)
                    elif current_section == "context":
                        contexts.append(item)
                    elif current_section == "task":
                        tasks.append(item)
        
        return mechanisms, contexts, tasks
    
    def extract_mct(self, idea_text: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract MCT triple from idea text using Ollama.
        
        Args:
            idea_text: The research idea text.
            
        Returns:
            Tuple of (mechanisms, contexts, tasks) lists.
        """
        prompt = self.MCT_EXTRACTION_PROMPT.format(idea_text=idea_text)
        
        try:
            response = self._call_ollama(prompt)
            mechanisms, contexts, tasks = self._parse_mct_response(response)
            
            if not mechanisms and not contexts and not tasks:
                logger.warning("Failed to extract any MCT components")
            
            return mechanisms, contexts, tasks
        except Exception as e:
            logger.error(f"MCT extraction failed: {e}")
            return [], [], []
    
    def compute_embeddings(
        self, 
        mechanisms: List[str],
        contexts: List[str],
        tasks: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute embeddings for MCT components.
        
        Args:
            mechanisms: List of mechanism names.
            contexts: List of context names.
            tasks: List of task names.
            
        Returns:
            Tuple of (mechanism_embeddings, context_embeddings, task_embeddings) dicts.
        """
        mechanism_embeddings = {}
        context_embeddings = {}
        task_embeddings = {}
        
        if mechanisms:
            embeddings = self.embedding_model.embed_texts(mechanisms)
            mechanism_embeddings = dict(zip(mechanisms, embeddings))
        
        if contexts:
            embeddings = self.embedding_model.embed_texts(contexts)
            context_embeddings = dict(zip(contexts, embeddings))
        
        if tasks:
            embeddings = self.embedding_model.embed_texts(tasks)
            task_embeddings = dict(zip(tasks, embeddings))
        
        return mechanism_embeddings, context_embeddings, task_embeddings
    
    def compute_parent_classes(
        self, 
        mechanisms: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
        """
        Compute parent classes and their embeddings for mechanisms.
        
        Args:
            mechanisms: List of mechanism names.
            
        Returns:
            Tuple of (parent_classes, parent_embeddings) dicts.
        """
        parent_classes = {}
        parent_embeddings = {}
        
        unique_parents = set()
        
        for mechanism in mechanisms:
            parent = get_parent_class(mechanism)
            parent_classes[mechanism] = parent
            unique_parents.add(parent)
        
        if unique_parents:
            parent_list = list(unique_parents)
            embeddings = self.embedding_model.embed_texts(parent_list)
            parent_embeddings = dict(zip(parent_list, embeddings))
        
        return parent_classes, parent_embeddings
    
    def process_idea(
        self, 
        idea_text: str,
        extract_mct: bool = True,
        compute_embeddings: bool = True
    ) -> MCTTriple:
        """
        Process an idea text to extract MCT and compute embeddings.
        
        Args:
            idea_text: The research idea text.
            extract_mct: Whether to use LLM to extract MCT (if False, idea_text should be MCT JSON).
            compute_embeddings: Whether to compute embeddings.
            
        Returns:
            MCTTriple with extracted components and embeddings.
        """
        if extract_mct:
            mechanisms, contexts, tasks = self.extract_mct(idea_text)
        else:
            # Parse idea_text as JSON
            try:
                data = json.loads(idea_text)
                mechanisms = data.get("mechanism", [])
                contexts = data.get("context", [])
                tasks = data.get("task", [])
            except json.JSONDecodeError:
                logger.error("Failed to parse idea_text as JSON")
                return MCTTriple([], [], [])
        
        result = MCTTriple(
            mechanisms=mechanisms,
            contexts=contexts,
            tasks=tasks
        )
        
        if compute_embeddings and (mechanisms or contexts or tasks):
            (result.mechanism_embeddings, 
             result.context_embeddings, 
             result.task_embeddings) = self.compute_embeddings(
                mechanisms, contexts, tasks
            )
            
            if mechanisms:
                (result.parent_classes, 
                 result.parent_embeddings) = self.compute_parent_classes(mechanisms)
        
        return result
    
    def process_mct_dict(
        self, 
        mct_dict: Dict[str, List[str]],
        compute_embeddings: bool = True
    ) -> MCTTriple:
        """
        Process a pre-extracted MCT dictionary.
        
        Args:
            mct_dict: Dictionary with 'mechanism', 'context', 'task' keys.
            compute_embeddings: Whether to compute embeddings.
            
        Returns:
            MCTTriple with components and embeddings.
        """
        mechanisms = mct_dict.get("mechanism", [])
        contexts = mct_dict.get("context", [])
        tasks = mct_dict.get("task", [])
        
        result = MCTTriple(
            mechanisms=mechanisms,
            contexts=contexts,
            tasks=tasks
        )
        
        if compute_embeddings and (mechanisms or contexts or tasks):
            (result.mechanism_embeddings, 
             result.context_embeddings, 
             result.task_embeddings) = self.compute_embeddings(
                mechanisms, contexts, tasks
            )
            
            if mechanisms:
                (result.parent_classes, 
                 result.parent_embeddings) = self.compute_parent_classes(mechanisms)
        
        return result
    
    def batch_process_ideas(
        self, 
        ideas: List[str],
        extract_mct: bool = True,
        compute_embeddings: bool = True
    ) -> List[MCTTriple]:
        """
        Process multiple ideas in batch.
        
        Args:
            ideas: List of idea texts.
            extract_mct: Whether to use LLM to extract MCT.
            compute_embeddings: Whether to compute embeddings.
            
        Returns:
            List of MCTTriple objects.
        """
        results = []
        
        for idea in ideas:
            result = self.process_idea(
                idea, 
                extract_mct=extract_mct,
                compute_embeddings=compute_embeddings
            )
            results.append(result)
        
        return results
