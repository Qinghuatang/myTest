"""
Embedding model module for Sci-Idea-EVAL system.
Provides SciBERT and Specter embedding functionality for MCT components.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding model for converting text to vectors using SciBERT or Specter.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding model.
        
        Args:
            config: Embedding configuration. Uses default if not provided.
        """
        self.config = config or EmbeddingConfig()
        self.device = torch.device(self.config.device)
        self.model = None
        self.tokenizer = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    def load_model(self) -> None:
        """Load the embedding model and tokenizer."""
        model_name = (
            self.config.specter_model if self.config.use_specter 
            else self.config.model_name
        )
        
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _mean_pooling(
        self, 
        model_output: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to get sentence embeddings.
        
        Args:
            model_output: Model output tensor.
            attention_mask: Attention mask tensor.
            
        Returns:
            Pooled embeddings tensor.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        if self.model is None:
            self.load_model()
        
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            output = self.model(**encoded)
            embedding = self._mean_pooling(output, encoded["attention_mask"])
            embedding = embedding.cpu().numpy()[0]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self._embedding_cache[text] = embedding
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Matrix of embeddings with shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        
        if self.model is None:
            self.load_model()
        
        # Check cache for existing embeddings
        cached_indices = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                cached_indices.append(i)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Process uncached texts in batches
        if uncached_texts:
            batch_size = self.config.batch_size
            new_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                
                with torch.no_grad():
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    output = self.model(**encoded)
                    batch_embeddings = self._mean_pooling(
                        output, encoded["attention_mask"]
                    )
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Normalize embeddings
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                batch_embeddings = batch_embeddings / norms
                
                new_embeddings.append(batch_embeddings)
                
                # Cache embeddings
                for j, text in enumerate(batch_texts):
                    self._embedding_cache[text] = batch_embeddings[j]
            
            new_embeddings = np.vstack(new_embeddings) if new_embeddings else np.array([])
        
        # Reconstruct full embedding matrix
        embeddings = np.zeros((len(texts), self.config.embedding_dim))
        
        for i, idx in enumerate(cached_indices):
            embeddings[idx] = self._embedding_cache[texts[idx]]
        
        for i, idx in enumerate(uncached_indices):
            embeddings[idx] = new_embeddings[i]
        
        return embeddings
    
    def save_cache(self, file_path: str) -> None:
        """
        Save embedding cache to file.
        
        Args:
            file_path: Path to save the cache.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        cache_data = {
            text: embedding.tolist() 
            for text, embedding in self._embedding_cache.items()
        }
        
        import json
        with open(file_path, "w") as f:
            json.dump(cache_data, f)
        
        logger.info(f"Saved {len(cache_data)} embeddings to {file_path}")
    
    def load_cache(self, file_path: str) -> None:
        """
        Load embedding cache from file.
        
        Args:
            file_path: Path to load the cache from.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Cache file not found: {file_path}")
            return
        
        import json
        with open(file_path, "r") as f:
            cache_data = json.load(f)
        
        self._embedding_cache = {
            text: np.array(embedding) 
            for text, embedding in cache_data.items()
        }
        
        logger.info(f"Loaded {len(self._embedding_cache)} embeddings from {file_path}")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.config.embedding_dim
