"""
Final scorer module for Sci-Idea-EVAL system.
Combines novelty, plausibility, and utility scores using gated product formula.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .config import Config, PlausibilityConfig
from .graph_builder import HeterogeneousGraphBuilder
from .input_processor import InputProcessor, MCTTriple
from .novelty_scorer import NoveltyScorer
from .plausibility_scorer import PlausibilityScorer
from .utility_scorer import UtilityScorer

logger = logging.getLogger(__name__)


@dataclass
class IdeaScore:
    """Container for idea evaluation scores."""
    final_score: float
    novelty_score: float
    plausibility_score: float
    utility_score: float
    is_plausible: bool
    details: Dict[str, Any]


class FinalScorer:
    """
    Computes final score for research ideas using gated product formula.
    
    Formula:
    Final_Score = S_Nov × S_Util × I(S_Plaus > θ)
    
    Where:
    - S_Nov: Novelty score
    - S_Util: Utility score
    - I(·): Indicator function
    - θ: Plausibility threshold (default 0.3)
    """
    
    def __init__(
        self,
        graph: HeterogeneousGraphBuilder,
        config: Optional[Config] = None,
        novelty_scorer: Optional[NoveltyScorer] = None,
        plausibility_scorer: Optional[PlausibilityScorer] = None,
        utility_scorer: Optional[UtilityScorer] = None,
        input_processor: Optional[InputProcessor] = None,
    ):
        """
        Initialize the final scorer.
        
        Args:
            graph: The heterogeneous MCT graph.
            config: Configuration object.
            novelty_scorer: Pre-initialized novelty scorer.
            plausibility_scorer: Pre-initialized plausibility scorer.
            utility_scorer: Pre-initialized utility scorer.
            input_processor: Pre-initialized input processor.
        """
        self.graph = graph
        self.config = config or Config()
        
        # Initialize scorers
        self.novelty_scorer = novelty_scorer or NoveltyScorer(
            graph, self.config.novelty
        )
        self.plausibility_scorer = plausibility_scorer or PlausibilityScorer(
            graph, self.config.link_prediction
        )
        self.utility_scorer = utility_scorer or UtilityScorer(
            graph, self.config.utility
        )
        self.input_processor = input_processor
        
        # Plausibility threshold
        self.threshold = self.config.plausibility.threshold
    
    def score_mct(self, mct_triple: MCTTriple) -> IdeaScore:
        """
        Score an MCT triple.
        
        Args:
            mct_triple: The extracted MCT triple with embeddings.
            
        Returns:
            IdeaScore with all component scores and final score.
        """
        # Compute individual scores
        novelty_score, novelty_details = self.novelty_scorer.score(mct_triple)
        plausibility_score, plausibility_details = self.plausibility_scorer.score(mct_triple)
        utility_score, utility_details = self.utility_scorer.score(mct_triple)
        
        # Check plausibility gate
        is_plausible = plausibility_score > self.threshold
        
        # Compute final score using gated product formula
        if is_plausible:
            final_score = novelty_score * utility_score
        else:
            final_score = 0.0
        
        details = {
            "novelty": novelty_details,
            "plausibility": plausibility_details,
            "utility": utility_details,
            "threshold": self.threshold,
            "mct": {
                "mechanisms": mct_triple.mechanisms,
                "contexts": mct_triple.contexts,
                "tasks": mct_triple.tasks,
            }
        }
        
        return IdeaScore(
            final_score=final_score,
            novelty_score=novelty_score,
            plausibility_score=plausibility_score,
            utility_score=utility_score,
            is_plausible=is_plausible,
            details=details,
        )
    
    def score_idea(
        self, 
        idea_text: str,
        extract_mct: bool = True
    ) -> IdeaScore:
        """
        Score a research idea from text.
        
        Args:
            idea_text: The research idea text or MCT JSON.
            extract_mct: Whether to extract MCT using LLM.
            
        Returns:
            IdeaScore with all component scores.
        """
        if self.input_processor is None:
            self.input_processor = InputProcessor(
                self.config, 
                self.graph.embedding_model
            )
        
        # Process the idea
        mct_triple = self.input_processor.process_idea(
            idea_text,
            extract_mct=extract_mct,
            compute_embeddings=True
        )
        
        return self.score_mct(mct_triple)
    
    def score_mct_dict(
        self, 
        mct_dict: Dict[str, list],
        compute_embeddings: bool = True
    ) -> IdeaScore:
        """
        Score an MCT dictionary directly.
        
        Args:
            mct_dict: Dictionary with 'mechanism', 'context', 'task' keys.
            compute_embeddings: Whether to compute embeddings.
            
        Returns:
            IdeaScore with all component scores.
        """
        if self.input_processor is None:
            self.input_processor = InputProcessor(
                self.config,
                self.graph.embedding_model
            )
        
        mct_triple = self.input_processor.process_mct_dict(
            mct_dict,
            compute_embeddings=compute_embeddings
        )
        
        return self.score_mct(mct_triple)
    
    def score_batch(
        self, 
        mct_triples: list,
    ) -> list:
        """
        Score multiple MCT triples.
        
        Args:
            mct_triples: List of MCTTriple objects.
            
        Returns:
            List of IdeaScore objects.
        """
        return [self.score_mct(mct) for mct in mct_triples]
    
    def compare_ideas(
        self, 
        mct_triples: list,
    ) -> Dict[str, Any]:
        """
        Compare multiple ideas and rank them.
        
        Args:
            mct_triples: List of MCTTriple objects.
            
        Returns:
            Dictionary with rankings and comparison details.
        """
        scores = self.score_batch(mct_triples)
        
        # Sort by final score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1].final_score,
            reverse=True
        )
        
        rankings = []
        for rank, (idx, score) in enumerate(ranked, 1):
            rankings.append({
                "rank": rank,
                "original_index": idx,
                "final_score": score.final_score,
                "novelty_score": score.novelty_score,
                "plausibility_score": score.plausibility_score,
                "utility_score": score.utility_score,
                "is_plausible": score.is_plausible,
            })
        
        return {
            "rankings": rankings,
            "best_idea_index": ranked[0][0] if ranked else None,
            "best_score": ranked[0][1].final_score if ranked else 0.0,
        }
    
    def get_score_breakdown(self, idea_score: IdeaScore) -> str:
        """
        Get a human-readable breakdown of the score.
        
        Args:
            idea_score: The IdeaScore to format.
            
        Returns:
            Formatted string with score breakdown.
        """
        plausibility_status = "✓" if idea_score.is_plausible else "✗"
        
        breakdown = f"""
=== Idea Evaluation Results ===
Final Score: {idea_score.final_score:.4f}

Component Scores:
  • Novelty (S_Nov):      {idea_score.novelty_score:.4f}
  • Plausibility (S_Plaus): {idea_score.plausibility_score:.4f} [{plausibility_status}]
  • Utility (S_Util):     {idea_score.utility_score:.4f}

Gate Status:
  • Threshold (θ):        {self.threshold}
  • Passes Gate:          {"Yes" if idea_score.is_plausible else "No"}

Formula: Final = S_Nov × S_Util × I(S_Plaus > θ)
       = {idea_score.novelty_score:.4f} × {idea_score.utility_score:.4f} × {1 if idea_score.is_plausible else 0}
       = {idea_score.final_score:.4f}
"""
        return breakdown
