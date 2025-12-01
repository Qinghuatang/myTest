"""
Validation pipeline module for Sci-Idea-EVAL system.
Implements retrospective validation experiments.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config, ValidationConfig
from .embedding_model import EmbeddingModel
from .final_scorer import FinalScorer, IdeaScore
from .graph_builder import HeterogeneousGraphBuilder
from .input_processor import InputProcessor, MCTTriple
from .novelty_scorer import NoveltyScorer
from .plausibility_scorer import PlausibilityScorer
from .utility_scorer import UtilityScorer
from .utils import (
    compute_hit_ratio_at_k,
    compute_spearman_correlation,
    extract_mct_from_paper,
    has_valid_mct,
    load_jsonl,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation experiment results."""
    ranking_accuracy: float
    hit_ratio_at_5: float
    hit_ratio_at_10: float
    spearman_correlation: float
    avg_score_gt: float
    avg_score_old: float
    avg_score_random: float
    details: Dict[str, Any]


class ValidationPipeline:
    """
    Validation pipeline for retrospective experiments.
    
    Workflow:
    1. Split data: 2018-2021 for training graph, 2022 for testing
    2. Build graph from training data
    3. Select positive samples: Top 10% cited 2022 papers
    4. Create negative samples:
       - Old: MCT from 2020 papers (stale ideas)
       - Random: Randomly shuffled M, C, T combinations
    5. Evaluate and compute metrics
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the validation pipeline.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.validation_config = self.config.validation
        
        # Components (initialized during run)
        self.graph: Optional[HeterogeneousGraphBuilder] = None
        self.embedding_model: Optional[EmbeddingModel] = None
        self.input_processor: Optional[InputProcessor] = None
        self.final_scorer: Optional[FinalScorer] = None
        
        # Data splits
        self.train_papers: List[Dict] = []
        self.test_papers: List[Dict] = []
        self.old_papers: List[Dict] = []
        
        # Samples
        self.positive_samples: List[MCTTriple] = []
        self.negative_old_samples: List[MCTTriple] = []
        self.negative_random_samples: List[MCTTriple] = []
    
    def load_and_split_data(self, data_path: str) -> None:
        """
        Load data and split by year.
        
        Args:
            data_path: Path to the JSONL data file.
        """
        logger.info(f"Loading data from {data_path}")
        
        train_years = set(self.validation_config.train_years)
        test_year = self.validation_config.test_year
        
        all_papers = list(load_jsonl(data_path))
        
        for paper in all_papers:
            year = paper.get("year", 0)
            
            if year in train_years and has_valid_mct(paper):
                self.train_papers.append(paper)
            elif year == test_year and has_valid_mct(paper):
                self.test_papers.append(paper)
            elif year == 2020 and has_valid_mct(paper):  # Old papers for negative samples
                self.old_papers.append(paper)
        
        logger.info(f"Train papers: {len(self.train_papers)}")
        logger.info(f"Test papers (2022): {len(self.test_papers)}")
        logger.info(f"Old papers (2020): {len(self.old_papers)}")
    
    def build_training_graph(self) -> None:
        """Build the graph from training data."""
        logger.info("Building training graph...")
        
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.graph = HeterogeneousGraphBuilder(
            self.config,
            self.embedding_model
        )
        
        # Build graph from train papers only
        # We need to save the papers temporarily and build from them
        import json
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for paper in self.train_papers:
                f.write(json.dumps(paper) + '\n')
            temp_path = f.name
        
        try:
            self.graph.build_from_data(temp_path, compute_embeddings=True)
        finally:
            import os
            os.unlink(temp_path)
        
        logger.info(f"Graph statistics: {self.graph.get_statistics()}")
    
    def prepare_samples(self) -> None:
        """Prepare positive and negative samples for validation."""
        logger.info("Preparing validation samples...")
        
        self.input_processor = InputProcessor(
            self.config,
            self.embedding_model
        )
        
        # Positive samples: Top 10% cited 2022 papers
        sorted_test = sorted(
            self.test_papers,
            key=lambda x: x.get("n_citation", 0),
            reverse=True
        )
        
        top_k = max(1, int(len(sorted_test) * self.validation_config.top_percentile))
        top_papers = sorted_test[:top_k]
        
        logger.info(f"Selected {len(top_papers)} top papers as positive samples")
        
        for paper in top_papers:
            mct_dict = paper.get("extracted_mct", {})
            mct_triple = self.input_processor.process_mct_dict(
                mct_dict, compute_embeddings=True
            )
            if mct_triple.mechanisms:
                self.positive_samples.append(mct_triple)
        
        # Negative samples A: Old papers (2020)
        num_old = min(self.validation_config.num_negative_old, len(self.old_papers))
        old_sample_papers = random.sample(self.old_papers, num_old)
        
        for paper in old_sample_papers:
            mct_dict = paper.get("extracted_mct", {})
            mct_triple = self.input_processor.process_mct_dict(
                mct_dict, compute_embeddings=True
            )
            if mct_triple.mechanisms:
                self.negative_old_samples.append(mct_triple)
        
        logger.info(f"Created {len(self.negative_old_samples)} old negative samples")
        
        # Negative samples B: Random shuffled MCT
        all_mechanisms = []
        all_contexts = []
        all_tasks = []
        
        for paper in self.train_papers[:500]:  # Use subset for efficiency
            m, c, t = extract_mct_from_paper(paper)
            all_mechanisms.extend(m)
            all_contexts.extend(c)
            all_tasks.extend(t)
        
        all_mechanisms = list(set(all_mechanisms))
        all_contexts = list(set(all_contexts))
        all_tasks = list(set(all_tasks))
        
        num_random = self.validation_config.num_negative_random
        
        for _ in range(num_random):
            # Random selection
            m = random.choice(all_mechanisms) if all_mechanisms else ""
            c = random.choice(all_contexts) if all_contexts else ""
            t = random.choice(all_tasks) if all_tasks else ""
            
            if m and c and t:
                mct_dict = {"mechanism": [m], "context": [c], "task": [t]}
                mct_triple = self.input_processor.process_mct_dict(
                    mct_dict, compute_embeddings=True
                )
                self.negative_random_samples.append(mct_triple)
        
        logger.info(f"Created {len(self.negative_random_samples)} random negative samples")
    
    def initialize_scorer(self, train_link_prediction: bool = True) -> None:
        """
        Initialize the final scorer.
        
        Args:
            train_link_prediction: Whether to train the link prediction model.
        """
        logger.info("Initializing scorer...")
        
        # Initialize plausibility scorer and train if needed
        plausibility_scorer = PlausibilityScorer(
            self.graph, self.config.link_prediction
        )
        
        if train_link_prediction:
            plausibility_scorer.train()
        
        self.final_scorer = FinalScorer(
            self.graph,
            self.config,
            plausibility_scorer=plausibility_scorer,
            input_processor=self.input_processor,
        )
    
    def evaluate_samples(self) -> Tuple[List[IdeaScore], List[IdeaScore], List[IdeaScore]]:
        """
        Evaluate all sample types.
        
        Returns:
            Tuple of (positive_scores, old_scores, random_scores).
        """
        logger.info("Evaluating samples...")
        
        positive_scores = []
        old_scores = []
        random_scores = []
        
        # Evaluate positive samples
        for mct in self.positive_samples:
            score = self.final_scorer.score_mct(mct)
            positive_scores.append(score)
        
        # Evaluate old samples
        for mct in self.negative_old_samples:
            score = self.final_scorer.score_mct(mct)
            old_scores.append(score)
        
        # Evaluate random samples
        for mct in self.negative_random_samples:
            score = self.final_scorer.score_mct(mct)
            random_scores.append(score)
        
        return positive_scores, old_scores, random_scores
    
    def compute_metrics(
        self,
        positive_scores: List[IdeaScore],
        old_scores: List[IdeaScore],
        random_scores: List[IdeaScore]
    ) -> ValidationResult:
        """
        Compute validation metrics.
        
        Args:
            positive_scores: Scores for positive (ground truth) samples.
            old_scores: Scores for old (stale) samples.
            random_scores: Scores for random (absurd) samples.
            
        Returns:
            ValidationResult with all metrics.
        """
        logger.info("Computing metrics...")
        
        # Extract final scores
        gt_scores = [s.final_score for s in positive_scores]
        old_scores_vals = [s.final_score for s in old_scores]
        random_scores_vals = [s.final_score for s in random_scores]
        
        avg_gt = np.mean(gt_scores) if gt_scores else 0.0
        avg_old = np.mean(old_scores_vals) if old_scores_vals else 0.0
        avg_random = np.mean(random_scores_vals) if random_scores_vals else 0.0
        
        # Ranking accuracy: % of times GT > Old > Random
        ranking_correct = 0
        total_comparisons = 0
        
        for gt in gt_scores:
            for old in old_scores_vals:
                for rand in random_scores_vals:
                    total_comparisons += 1
                    if gt > old > rand:
                        ranking_correct += 1
        
        ranking_accuracy = ranking_correct / total_comparisons if total_comparisons > 0 else 0.0
        
        # Hit Ratio @ K
        all_scores = []
        gt_ids = set()
        
        for i, score in enumerate(positive_scores):
            all_scores.append((f"gt_{i}", score.final_score))
            gt_ids.add(f"gt_{i}")
        
        for i, score in enumerate(old_scores):
            all_scores.append((f"old_{i}", score.final_score))
        
        for i, score in enumerate(random_scores):
            all_scores.append((f"rand_{i}", score.final_score))
        
        # Sort by score descending
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        hr_at_5 = compute_hit_ratio_at_k(all_scores, gt_ids, 5)
        hr_at_10 = compute_hit_ratio_at_k(all_scores, gt_ids, 10)
        
        # Spearman correlation with citations (for GT papers only)
        gt_citations = [
            p.get("n_citation", 0) 
            for p in sorted(
                self.test_papers,
                key=lambda x: x.get("n_citation", 0),
                reverse=True
            )[:len(positive_scores)]
        ]
        
        spearman = compute_spearman_correlation(gt_scores, gt_citations)
        
        details = {
            "num_positive": len(positive_scores),
            "num_old": len(old_scores),
            "num_random": len(random_scores),
            "score_distribution": {
                "gt_mean": float(avg_gt),
                "gt_std": float(np.std(gt_scores)) if gt_scores else 0.0,
                "old_mean": float(avg_old),
                "old_std": float(np.std(old_scores_vals)) if old_scores_vals else 0.0,
                "random_mean": float(avg_random),
                "random_std": float(np.std(random_scores_vals)) if random_scores_vals else 0.0,
            }
        }
        
        return ValidationResult(
            ranking_accuracy=float(ranking_accuracy),
            hit_ratio_at_5=float(hr_at_5),
            hit_ratio_at_10=float(hr_at_10),
            spearman_correlation=float(spearman),
            avg_score_gt=float(avg_gt),
            avg_score_old=float(avg_old),
            avg_score_random=float(avg_random),
            details=details,
        )
    
    def run(
        self, 
        data_path: str,
        train_link_prediction: bool = True
    ) -> ValidationResult:
        """
        Run the full validation pipeline.
        
        Args:
            data_path: Path to the JSONL data file.
            train_link_prediction: Whether to train link prediction model.
            
        Returns:
            ValidationResult with all metrics.
        """
        logger.info("Starting validation pipeline...")
        
        # Step 1: Load and split data
        self.load_and_split_data(data_path)
        
        # Step 2: Build training graph
        self.build_training_graph()
        
        # Step 3: Prepare samples
        self.prepare_samples()
        
        # Step 4: Initialize scorer
        self.initialize_scorer(train_link_prediction)
        
        # Step 5: Evaluate samples
        positive_scores, old_scores, random_scores = self.evaluate_samples()
        
        # Step 6: Compute metrics
        result = self.compute_metrics(positive_scores, old_scores, random_scores)
        
        logger.info("Validation pipeline completed")
        self._print_results(result)
        
        return result
    
    def _print_results(self, result: ValidationResult) -> None:
        """Print validation results in a formatted way."""
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)
        print(f"\nRanking Metrics:")
        print(f"  • Ranking Accuracy (GT > Old > Random): {result.ranking_accuracy:.4f}")
        print(f"  • Hit Ratio @ 5: {result.hit_ratio_at_5:.4f}")
        print(f"  • Hit Ratio @ 10: {result.hit_ratio_at_10:.4f}")
        print(f"  • Spearman Correlation with Citations: {result.spearman_correlation:.4f}")
        print(f"\nAverage Scores:")
        print(f"  • Ground Truth (2022 Top Papers): {result.avg_score_gt:.4f}")
        print(f"  • Old Papers (2020): {result.avg_score_old:.4f}")
        print(f"  • Random Combinations: {result.avg_score_random:.4f}")
        print(f"\nExpected Ordering: GT > Old > Random")
        print(f"Actual Ordering: {result.avg_score_gt:.4f} > {result.avg_score_old:.4f} > {result.avg_score_random:.4f}")
        print("=" * 50 + "\n")
