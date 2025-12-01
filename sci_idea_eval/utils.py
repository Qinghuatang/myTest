"""
Utility functions for Sci-Idea-EVAL system.
"""

import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file.
        
    Yields:
        Dictionary for each line in the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save.
        file_path: Path to the output file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity value between -1 and 1.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine distance (1 - cosine similarity) between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine distance value between 0 and 2.
    """
    return 1 - cosine_similarity(vec1, vec2)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to [0, 1] range.
    
    Args:
        score: Score to normalize.
        min_val: Minimum value in the original range.
        max_val: Maximum value in the original range.
        
    Returns:
        Normalized score in [0, 1] range.
    """
    if max_val == min_val:
        return 0.5
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and multiple vectors.
    
    Args:
        query: Query vector of shape (dim,).
        vectors: Matrix of vectors of shape (n, dim).
        
    Returns:
        Array of cosine similarities of shape (n,).
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(vectors))
    
    query_normalized = query / query_norm
    
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vector_norms = np.where(vector_norms == 0, 1, vector_norms)  # Avoid division by zero
    vectors_normalized = vectors / vector_norms
    
    return np.dot(vectors_normalized, query_normalized)


def extract_mct_from_paper(paper: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract MCT (Mechanism, Context, Task) from a paper dictionary.
    
    Args:
        paper: Paper dictionary with 'extracted_mct' field.
        
    Returns:
        Tuple of (mechanisms, contexts, tasks) lists.
    """
    mct = paper.get("extracted_mct", {})
    mechanisms = mct.get("mechanism", [])
    contexts = mct.get("context", [])
    tasks = mct.get("task", [])
    
    return mechanisms, contexts, tasks


def has_valid_mct(paper: Dict[str, Any]) -> bool:
    """
    Check if a paper has valid (non-empty) MCT fields.
    
    Args:
        paper: Paper dictionary.
        
    Returns:
        True if all MCT fields are non-empty.
    """
    mechanisms, contexts, tasks = extract_mct_from_paper(paper)
    return bool(mechanisms) and bool(contexts) and bool(tasks)


def get_parent_class(mechanism: str, parent_mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Get the parent class of a mechanism.
    Uses the first word or phrase before common delimiters as a heuristic.
    
    Args:
        mechanism: Mechanism name.
        parent_mapping: Optional pre-defined mapping of mechanisms to parent classes.
        
    Returns:
        Parent class name.
    """
    if parent_mapping and mechanism in parent_mapping:
        return parent_mapping[mechanism]
    
    # Simple heuristic: use first significant word/phrase
    # Remove common suffixes
    suffixes = [" Model", " Method", " Algorithm", " Network", " Mechanism", " Framework"]
    parent = mechanism
    for suffix in suffixes:
        if parent.endswith(suffix):
            parent = parent[:-len(suffix)]
            break
    
    # If still long, take first few words
    words = parent.split()
    if len(words) > 3:
        parent = " ".join(words[:3])
    
    return parent if parent else mechanism


def compute_spearman_correlation(scores: List[float], citations: List[int]) -> float:
    """
    Compute Spearman's rank correlation coefficient.
    
    Args:
        scores: List of predicted scores.
        citations: List of citation counts.
        
    Returns:
        Spearman correlation coefficient.
    """
    from scipy import stats
    
    if len(scores) < 2:
        return 0.0
    
    correlation, _ = stats.spearmanr(scores, citations)
    return float(correlation) if not np.isnan(correlation) else 0.0


def compute_hit_ratio_at_k(ranked_scores: List[Tuple[str, float]], 
                           ground_truth: set, 
                           k: int) -> float:
    """
    Compute Hit Ratio @ K.
    
    Args:
        ranked_scores: List of (id, score) tuples sorted by score descending.
        ground_truth: Set of ground truth item IDs.
        k: Number of top items to consider.
        
    Returns:
        Hit ratio @ K value.
    """
    if not ground_truth or k <= 0:
        return 0.0
    
    top_k_ids = {item[0] for item in ranked_scores[:k]}
    hits = len(top_k_ids & ground_truth)
    
    return hits / min(k, len(ground_truth))


def save_numpy_array(array: np.ndarray, file_path: str) -> None:
    """Save numpy array to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, array)


def load_numpy_array(file_path: str) -> np.ndarray:
    """Load numpy array from file."""
    return np.load(file_path)


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
