"""
Sci-Idea-EVAL (MCT-Edition)
A benchmark system for automatically evaluating the quality of LLM-generated research ideas.

This package implements novelty, plausibility, and utility scoring for scientific ideas
based on Mechanism-Context-Task (MCT) triple extraction and heterogeneous graph analysis.
"""

from .config import Config
from .graph_builder import HeterogeneousGraphBuilder
from .input_processor import InputProcessor
from .embedding_model import EmbeddingModel
from .novelty_scorer import NoveltyScorer
from .plausibility_scorer import PlausibilityScorer
from .utility_scorer import UtilityScorer
from .final_scorer import FinalScorer
from .validation_pipeline import ValidationPipeline

__version__ = "1.0.0"
__all__ = [
    "Config",
    "HeterogeneousGraphBuilder",
    "InputProcessor",
    "EmbeddingModel",
    "NoveltyScorer",
    "PlausibilityScorer",
    "UtilityScorer",
    "FinalScorer",
    "ValidationPipeline",
]
