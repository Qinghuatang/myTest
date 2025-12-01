"""
Utility scorer module for Sci-Idea-EVAL system.
Computes utility score based on task heat and mechanism-context fit.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config, UtilityConfig
from .graph_builder import HeterogeneousGraphBuilder
from .input_processor import MCTTriple
from .utils import cosine_similarity, normalize_score

logger = logging.getLogger(__name__)


class UtilityScorer:
    """
    Computes utility score based on task heat and demand matching.
    
    Utility measures:
    1. Task Heat: How "hot" is the target task based on citation-weighted activity
    2. Demand Fit: How well does the mechanism match the requirements of the context
    
    Formula:
    S_Util = Normalize(Heat(t)) * Fit(m, c)
    
    Where:
    - Heat(t) = Σ log(p.n_citation + 1) / max(1, 2023 - p.year) for papers solving task t
    - Fit(m, c) = CosineSim(Vec_Requirement(c), Vec_Function(m))
    """
    
    def __init__(
        self, 
        graph: HeterogeneousGraphBuilder,
        config: Optional[UtilityConfig] = None
    ):
        """
        Initialize the utility scorer.
        
        Args:
            graph: The heterogeneous MCT graph.
            config: Utility scoring configuration.
        """
        self.graph = graph
        self.config = config or UtilityConfig()
        
        # Cache for task heat values
        self._heat_cache: Dict[str, float] = {}
        self._max_heat: float = 0.0
        self._min_heat: float = 0.0
        
        # Precompute heat values
        self._precompute_heat_values()
    
    def _precompute_heat_values(self) -> None:
        """Precompute heat values for all tasks in the graph."""
        all_heats = []
        
        for task_id in self.graph.tasks:
            heat = self._compute_raw_heat(task_id)
            self._heat_cache[task_id] = heat
            all_heats.append(heat)
        
        if all_heats:
            self._max_heat = max(all_heats)
            self._min_heat = min(all_heats)
        
        logger.info(f"Precomputed heat values for {len(self._heat_cache)} tasks")
        logger.info(f"Heat range: [{self._min_heat:.2f}, {self._max_heat:.2f}]")
    
    def _compute_raw_heat(self, task_id: str) -> float:
        """
        Compute raw heat value for a task.
        
        Heat(t) = Σ log(p.n_citation + 1) / max(1, reference_year - p.year)
        
        Args:
            task_id: Task node ID.
            
        Returns:
            Raw heat value.
        """
        task_name = task_id.split(":", 1)[1] if ":" in task_id else task_id
        neighbors = self.graph.get_task_neighbors(task_name)
        
        if not neighbors:
            return 0.0
        
        heat = 0.0
        reference_year = self.config.reference_year
        
        for paper in neighbors:
            n_citation = paper.get("n_citation", 0)
            year = paper.get("year", reference_year - 1)
            
            # Citation impact
            citation_score = math.log(n_citation + 1)
            
            # Recency weight
            years_ago = max(1, reference_year - year)
            recency_weight = 1.0 / years_ago
            
            heat += citation_score * recency_weight
        
        return heat
    
    def compute_task_heat(self, task: str) -> float:
        """
        Compute normalized heat value for a task.
        
        Args:
            task: Task name.
            
        Returns:
            Normalized heat value in [0, 1].
        """
        task_id = f"task:{task.strip().lower()}"
        
        # Check cache first
        if task_id in self._heat_cache:
            raw_heat = self._heat_cache[task_id]
        else:
            raw_heat = self._compute_raw_heat(task_id)
            self._heat_cache[task_id] = raw_heat
        
        # Normalize
        if self._max_heat == self._min_heat:
            return 0.5
        
        return normalize_score(raw_heat, self._min_heat, self._max_heat)
    
    def compute_demand_fit(
        self, 
        mechanism_embedding: np.ndarray,
        context_embedding: np.ndarray
    ) -> float:
        """
        Compute demand fit between mechanism and context.
        
        Uses cosine similarity between mechanism embedding (representing function)
        and context embedding (representing requirements).
        
        Args:
            mechanism_embedding: Embedding of the mechanism.
            context_embedding: Embedding of the context.
            
        Returns:
            Fit score in [0, 1].
        """
        similarity = cosine_similarity(mechanism_embedding, context_embedding)
        
        # Convert similarity from [-1, 1] to [0, 1]
        fit = (similarity + 1) / 2
        
        return float(fit)
    
    def score_single_combination(
        self,
        mechanism: str,
        context: str,
        task: str,
        mechanism_embedding: Optional[np.ndarray] = None,
        context_embedding: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        Score utility for a single M-C-T combination.
        
        Args:
            mechanism: Mechanism name.
            context: Context name.
            task: Task name.
            mechanism_embedding: Optional pre-computed mechanism embedding.
            context_embedding: Optional pre-computed context embedding.
            
        Returns:
            Tuple of (utility_score, details_dict).
        """
        # Compute task heat
        heat = self.compute_task_heat(task)
        
        details = {
            "mechanism": mechanism,
            "context": context,
            "task": task,
            "task_heat": heat,
        }
        
        # Compute demand fit
        if mechanism_embedding is not None and context_embedding is not None:
            fit = self.compute_demand_fit(mechanism_embedding, context_embedding)
        else:
            # Try to get embeddings from graph
            mech_id = f"mechanism:{mechanism.strip().lower()}"
            ctx_id = f"context:{context.strip().lower()}"
            
            mech_emb = self.graph.get_mechanism_embedding(mech_id)
            ctx_emb = self.graph.get_context_embedding(ctx_id)
            
            if mech_emb is not None and ctx_emb is not None:
                fit = self.compute_demand_fit(mech_emb, ctx_emb)
            else:
                # Default fit if embeddings not available
                fit = self.config.default_fit_score if hasattr(self.config, 'default_fit_score') else 0.5
                details["fit_fallback"] = True
        
        details["demand_fit"] = fit
        
        # Final utility score
        utility = heat * fit
        
        return utility, details
    
    def score(self, mct_triple: MCTTriple) -> Tuple[float, Dict]:
        """
        Score utility for an MCT triple.
        
        Args:
            mct_triple: The extracted MCT triple with embeddings.
            
        Returns:
            Tuple of (utility_score, details_dict).
        """
        if not mct_triple.mechanisms:
            logger.warning("No mechanisms in MCT triple")
            return 0.0, {"error": "no_mechanisms"}
        
        if not mct_triple.contexts or not mct_triple.tasks:
            logger.warning("Missing contexts or tasks")
            return 0.0, {"error": "missing_context_or_task"}
        
        scores = []
        all_details = []
        
        for mechanism in mct_triple.mechanisms:
            mech_embedding = None
            if mct_triple.mechanism_embeddings:
                mech_embedding = mct_triple.mechanism_embeddings.get(mechanism)
            
            for context in mct_triple.contexts:
                ctx_embedding = None
                if mct_triple.context_embeddings:
                    ctx_embedding = mct_triple.context_embeddings.get(context)
                
                for task in mct_triple.tasks:
                    score, details = self.score_single_combination(
                        mechanism, context, task,
                        mechanism_embedding=mech_embedding,
                        context_embedding=ctx_embedding
                    )
                    scores.append(score)
                    all_details.append(details)
        
        if not scores:
            return 0.0, {"error": "no_valid_combinations"}
        
        avg_score = float(np.mean(scores))
        
        return avg_score, {
            "avg_utility": avg_score,
            "num_combinations": len(scores),
            "combination_details": all_details,
        }
    
    def get_top_hot_tasks(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top k hottest tasks.
        
        Args:
            k: Number of tasks to return.
            
        Returns:
            List of (task_name, heat_value) tuples sorted by heat.
        """
        sorted_tasks = sorted(
            self._heat_cache.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Convert task_id to task_name
        return [
            (task_id.split(":", 1)[1] if ":" in task_id else task_id, heat)
            for task_id, heat in sorted_tasks
        ]
    
    def get_heat_statistics(self) -> Dict[str, float]:
        """Get statistics about task heat distribution."""
        heats = list(self._heat_cache.values())
        
        if not heats:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "num_tasks": 0,
            }
        
        return {
            "min": float(np.min(heats)),
            "max": float(np.max(heats)),
            "mean": float(np.mean(heats)),
            "std": float(np.std(heats)),
            "num_tasks": len(heats),
        }
