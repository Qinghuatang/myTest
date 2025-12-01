"""
Sci-Idea-EVAL (MCT-Edition)
A benchmark system for automatically evaluating the quality of LLM-generated research ideas.

This package implements novelty, plausibility, and utility scoring for scientific ideas
based on Mechanism-Context-Task (MCT) triple extraction and heterogeneous graph analysis.
"""

__version__ = "1.0.0"

# Lazy imports to avoid dependency errors
def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "Config":
        from .config import Config
        return Config
    elif name == "HeterogeneousGraphBuilder":
        from .graph_builder import HeterogeneousGraphBuilder
        return HeterogeneousGraphBuilder
    elif name == "InputProcessor":
        from .input_processor import InputProcessor
        return InputProcessor
    elif name == "EmbeddingModel":
        from .embedding_model import EmbeddingModel
        return EmbeddingModel
    elif name == "NoveltyScorer":
        from .novelty_scorer import NoveltyScorer
        return NoveltyScorer
    elif name == "PlausibilityScorer":
        from .plausibility_scorer import PlausibilityScorer
        return PlausibilityScorer
    elif name == "UtilityScorer":
        from .utility_scorer import UtilityScorer
        return UtilityScorer
    elif name == "FinalScorer":
        from .final_scorer import FinalScorer
        return FinalScorer
    elif name == "ValidationPipeline":
        from .validation_pipeline import ValidationPipeline
        return ValidationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
