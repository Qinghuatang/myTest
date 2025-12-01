"""
Configuration module for Sci-Idea-EVAL system.
Contains model configurations, paths, and hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM API."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:70b"
    timeout: int = 120
    max_retries: int = 3


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "allenai/scibert_scivocab_uncased"
    specter_model: str = "allenai/specter2"
    embedding_dim: int = 768
    use_specter: bool = False  # If True, use Specter; otherwise use SciBERT
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class GraphConfig:
    """Configuration for heterogeneous graph construction."""
    cache_dir: str = "cache/graph"
    save_embeddings: bool = True


@dataclass
class LinkPredictionConfig:
    """Configuration for link prediction model (PyKEEN)."""
    model_name: str = "RotatE"  # Options: "RotatE", "ComplEx"
    embedding_dim: int = 128
    num_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 256
    num_negatives: int = 10
    model_save_path: str = "cache/models/link_prediction"


@dataclass
class NoveltyConfig:
    """Configuration for novelty scoring."""
    alpha: float = 0.7  # Weight for minimum semantic distance
    isolation_threshold: float = 0.15  # Threshold for isolation penalty
    isolation_penalty: float = 0.0  # Score when isolated
    default_empty_history_score: float = 0.5  # Score when no historical data exists


@dataclass
class PlausibilityConfig:
    """Configuration for plausibility scoring."""
    threshold: float = 0.3  # Threshold θ for gate function
    unknown_entity_score: float = 0.3  # Score for unknown entities
    max_samples_per_combination: int = 5  # Max samples for scoring each combination


@dataclass
class UtilityConfig:
    """Configuration for utility scoring."""
    reference_year: int = 2023  # Reference year for heat calculation
    default_fit_score: float = 0.5  # Default fit score when embeddings unavailable


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    train_years: list = field(default_factory=lambda: [2018, 2019, 2020, 2021])
    test_year: int = 2022
    old_papers_year: int = 2020  # Year for old negative samples
    top_percentile: float = 0.1  # Top 10% papers for testing
    num_negative_old: int = 100  # Number of old negative samples
    num_negative_random: int = 100  # Number of random negative samples
    random_sample_subset_size: int = 500  # Subset size for random sample generation


@dataclass
class PathConfig:
    """Path configurations."""
    data_file: str = "mct_results_v2.jsonl"
    cache_dir: str = "cache"
    models_dir: str = "cache/models"
    embeddings_dir: str = "cache/embeddings"
    graph_dir: str = "cache/graph"


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    link_prediction: LinkPredictionConfig = field(default_factory=LinkPredictionConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    plausibility: PlausibilityConfig = field(default_factory=PlausibilityConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    def __post_init__(self):
        """Create cache directories if they don't exist."""
        os.makedirs(self.paths.cache_dir, exist_ok=True)
        os.makedirs(self.paths.models_dir, exist_ok=True)
        os.makedirs(self.paths.embeddings_dir, exist_ok=True)
        os.makedirs(self.paths.graph_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        ollama = OllamaConfig(**config_dict.get("ollama", {}))
        embedding = EmbeddingConfig(**config_dict.get("embedding", {}))
        graph = GraphConfig(**config_dict.get("graph", {}))
        link_prediction = LinkPredictionConfig(**config_dict.get("link_prediction", {}))
        novelty = NoveltyConfig(**config_dict.get("novelty", {}))
        plausibility = PlausibilityConfig(**config_dict.get("plausibility", {}))
        utility = UtilityConfig(**config_dict.get("utility", {}))
        validation = ValidationConfig(**config_dict.get("validation", {}))
        paths = PathConfig(**config_dict.get("paths", {}))
        
        return cls(
            ollama=ollama,
            embedding=embedding,
            graph=graph,
            link_prediction=link_prediction,
            novelty=novelty,
            plausibility=plausibility,
            utility=utility,
            validation=validation,
            paths=paths,
        )
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
