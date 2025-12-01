"""
Novelty scorer module for Sci-Idea-EVAL system.
Computes novelty score based on semantic distance in MCT graph.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import Config, NoveltyConfig
from .graph_builder import HeterogeneousGraphBuilder
from .input_processor import MCTTriple
from .utils import batch_cosine_similarity, cosine_similarity, normalize_score

logger = logging.getLogger(__name__)


class NoveltyScorer:
    """
    Computes novelty score for research ideas based on semantic jump distance.
    
    Novelty is defined as the semantic distance of the proposed mechanism
    from historically used mechanisms within the same context-task intersection.
    
    Formula:
    S_Nov = α * D_min + (1-α) * D_parent
    
    Where:
    - D_min: Minimum semantic distance to any historical mechanism
    - D_parent: Minimum semantic distance between parent classes
    - α: Weight parameter (default 0.7)
    """
    
    def __init__(
        self, 
        graph: HeterogeneousGraphBuilder,
        config: Optional[NoveltyConfig] = None
    ):
        """
        Initialize the novelty scorer.
        
        Args:
            graph: The heterogeneous MCT graph.
            config: Novelty scoring configuration.
        """
        self.graph = graph
        self.config = config or NoveltyConfig()
        
        # Cache for historical mechanism embeddings
        self._history_cache: Dict[Tuple[str, str], List[np.ndarray]] = {}
    
    def _get_historical_mechanisms(
        self, 
        context: str, 
        task: str
    ) -> Tuple[Set[str], List[np.ndarray]]:
        """
        Get historical mechanisms for a context-task pair.
        
        Args:
            context: Context name.
            task: Task name.
            
        Returns:
            Tuple of (mechanism_ids, mechanism_embeddings).
        """
        # Find papers with both context and task
        anchor_papers = self.graph.get_papers_with_context_and_task(context, task)
        
        if not anchor_papers:
            logger.debug(f"No anchor papers found for context='{context}', task='{task}'")
            return set(), []
        
        # Get mechanisms from these papers
        mechanism_ids = self.graph.get_mechanisms_for_papers(anchor_papers)
        
        # Get embeddings
        embeddings = []
        valid_ids = set()
        
        for mech_id in mechanism_ids:
            emb = self.graph.get_mechanism_embedding(mech_id)
            if emb is not None:
                embeddings.append(emb)
                valid_ids.add(mech_id)
        
        return valid_ids, embeddings
    
    def _get_historical_parent_embeddings(
        self, 
        mechanism_ids: Set[str]
    ) -> List[np.ndarray]:
        """
        Get parent class embeddings for a set of mechanisms.
        
        Args:
            mechanism_ids: Set of mechanism node IDs.
            
        Returns:
            List of parent class embeddings.
        """
        embeddings = []
        seen_parents = set()
        
        for mech_id in mechanism_ids:
            parent_emb = self.graph.get_parent_embedding(mech_id)
            parent_class = self.graph.parent_classes.get(mech_id)
            
            if parent_emb is not None and parent_class not in seen_parents:
                embeddings.append(parent_emb)
                seen_parents.add(parent_class)
        
        return embeddings
    
    def _compute_min_distance(
        self, 
        query_embedding: np.ndarray, 
        reference_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute minimum semantic distance from query to reference embeddings.
        
        Args:
            query_embedding: Query vector.
            reference_embeddings: List of reference vectors.
            
        Returns:
            Minimum distance (1 - max_similarity).
        """
        if not reference_embeddings:
            return 1.0  # Maximum distance if no references
        
        reference_matrix = np.array(reference_embeddings)
        similarities = batch_cosine_similarity(query_embedding, reference_matrix)
        max_similarity = float(np.max(similarities))
        
        return 1.0 - max_similarity
    
    def _check_isolation(
        self, 
        mechanism_embedding: np.ndarray, 
        all_mechanism_embeddings: List[np.ndarray]
    ) -> bool:
        """
        Check if a mechanism is isolated (very low similarity to all known mechanisms).
        
        Args:
            mechanism_embedding: The new mechanism's embedding.
            all_mechanism_embeddings: All known mechanism embeddings.
            
        Returns:
            True if the mechanism is isolated.
        """
        if not all_mechanism_embeddings:
            return True
        
        reference_matrix = np.array(all_mechanism_embeddings)
        similarities = batch_cosine_similarity(mechanism_embedding, reference_matrix)
        max_similarity = float(np.max(similarities))
        
        return max_similarity < self.config.isolation_threshold
    
    def score_single_mechanism(
        self, 
        mechanism_embedding: np.ndarray,
        parent_embedding: Optional[np.ndarray],
        context: str,
        task: str
    ) -> Tuple[float, Dict]:
        """
        Score novelty for a single mechanism in a context-task setting.
        
        Args:
            mechanism_embedding: Embedding of the new mechanism.
            parent_embedding: Embedding of the mechanism's parent class.
            context: Context name.
            task: Task name.
            
        Returns:
            Tuple of (novelty_score, details_dict).
        """
        # Get historical mechanisms
        history_ids, history_embeddings = self._get_historical_mechanisms(context, task)
        
        details = {
            "num_historical_mechanisms": len(history_ids),
            "context": context,
            "task": task,
        }
        
        # Handle empty history
        if not history_embeddings:
            # No prior work in this context-task - moderate novelty
            details["d_min"] = 0.5
            details["d_parent"] = 0.5
            details["empty_history"] = True
            return 0.5, details
        
        # Compute minimum semantic distance
        d_min = self._compute_min_distance(mechanism_embedding, history_embeddings)
        details["d_min"] = d_min
        
        # Compute parent class distance
        d_parent = 0.5  # Default if no parent embedding
        if parent_embedding is not None:
            parent_embeddings = self._get_historical_parent_embeddings(history_ids)
            if parent_embeddings:
                d_parent = self._compute_min_distance(parent_embedding, parent_embeddings)
        details["d_parent"] = d_parent
        
        # Check for isolation penalty
        all_mechanism_embeddings = [
            emb for emb in (self.graph.get_mechanism_embedding(m) 
                           for m in self.graph.mechanisms)
            if emb is not None
        ]
        
        if self._check_isolation(mechanism_embedding, all_mechanism_embeddings):
            details["isolated"] = True
            return self.config.isolation_penalty, details
        
        details["isolated"] = False
        
        # Compute final novelty score
        novelty = self.config.alpha * d_min + (1 - self.config.alpha) * d_parent
        
        # Normalize to [0, 1]
        novelty = normalize_score(novelty, 0.0, 1.0)
        
        return novelty, details
    
    def score(self, mct_triple: MCTTriple) -> Tuple[float, Dict]:
        """
        Score novelty for an MCT triple.
        
        Computes novelty for each (mechanism, context, task) combination
        and returns the average score.
        
        Args:
            mct_triple: The extracted MCT triple with embeddings.
            
        Returns:
            Tuple of (novelty_score, details_dict).
        """
        if not mct_triple.mechanisms:
            logger.warning("No mechanisms in MCT triple")
            return 0.0, {"error": "no_mechanisms"}
        
        if not mct_triple.contexts or not mct_triple.tasks:
            logger.warning("Missing contexts or tasks in MCT triple")
            return 0.0, {"error": "missing_context_or_task"}
        
        if not mct_triple.mechanism_embeddings:
            logger.warning("No mechanism embeddings computed")
            return 0.0, {"error": "no_embeddings"}
        
        scores = []
        all_details = []
        
        for mechanism in mct_triple.mechanisms:
            mech_embedding = mct_triple.mechanism_embeddings.get(mechanism)
            if mech_embedding is None:
                continue
            
            parent_class = (mct_triple.parent_classes or {}).get(mechanism)
            parent_embedding = None
            if parent_class and mct_triple.parent_embeddings:
                parent_embedding = mct_triple.parent_embeddings.get(parent_class)
            
            for context in mct_triple.contexts:
                for task in mct_triple.tasks:
                    score, details = self.score_single_mechanism(
                        mech_embedding, parent_embedding, context, task
                    )
                    details["mechanism"] = mechanism
                    scores.append(score)
                    all_details.append(details)
        
        if not scores:
            return 0.0, {"error": "no_valid_combinations"}
        
        avg_score = float(np.mean(scores))
        
        return avg_score, {
            "avg_novelty": avg_score,
            "num_combinations": len(scores),
            "combination_details": all_details,
        }
    
    def score_from_embeddings(
        self,
        mechanism_embeddings: Dict[str, np.ndarray],
        context_embeddings: Dict[str, np.ndarray],
        task_embeddings: Dict[str, np.ndarray],
        parent_classes: Optional[Dict[str, str]] = None,
        parent_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[float, Dict]:
        """
        Score novelty directly from embedding dictionaries.
        
        Args:
            mechanism_embeddings: Dict mapping mechanism names to embeddings.
            context_embeddings: Dict mapping context names to embeddings.
            task_embeddings: Dict mapping task names to embeddings.
            parent_classes: Optional dict mapping mechanisms to parent classes.
            parent_embeddings: Optional dict mapping parent classes to embeddings.
            
        Returns:
            Tuple of (novelty_score, details_dict).
        """
        mct_triple = MCTTriple(
            mechanisms=list(mechanism_embeddings.keys()),
            contexts=list(context_embeddings.keys()),
            tasks=list(task_embeddings.keys()),
            mechanism_embeddings=mechanism_embeddings,
            context_embeddings=context_embeddings,
            task_embeddings=task_embeddings,
            parent_classes=parent_classes,
            parent_embeddings=parent_embeddings,
        )
        
        return self.score(mct_triple)
