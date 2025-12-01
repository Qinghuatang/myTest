#!/usr/bin/env python3
"""
Sci-Idea-EVAL (MCT-Edition) - Main Entry Point

A benchmark system for automatically evaluating the quality of LLM-generated
research ideas based on novelty, plausibility, and utility.

Usage:
    python -m sci_idea_eval.main build-graph --data-path mct_results_v2.jsonl
    python -m sci_idea_eval.main train-lp --data-path mct_results_v2.jsonl
    python -m sci_idea_eval.main evaluate --idea "Your research idea here"
    python -m sci_idea_eval.main validate --data-path mct_results_v2.jsonl
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install them with: pip install -r requirements.txt")
        sys.exit(1)


def _import_modules():
    """Import modules after dependency check."""
    from .config import Config
    from .embedding_model import EmbeddingModel
    from .final_scorer import FinalScorer
    from .graph_builder import HeterogeneousGraphBuilder
    from .input_processor import InputProcessor
    from .plausibility_scorer import PlausibilityScorer
    from .validation_pipeline import ValidationPipeline
    return {
        'Config': Config,
        'EmbeddingModel': EmbeddingModel,
        'FinalScorer': FinalScorer,
        'HeterogeneousGraphBuilder': HeterogeneousGraphBuilder,
        'InputProcessor': InputProcessor,
        'PlausibilityScorer': PlausibilityScorer,
        'ValidationPipeline': ValidationPipeline,
    }


def build_graph(args: argparse.Namespace) -> None:
    """Build the heterogeneous MCT graph from data."""
    _check_dependencies()
    modules = _import_modules()
    
    Config = modules['Config']
    EmbeddingModel = modules['EmbeddingModel']
    HeterogeneousGraphBuilder = modules['HeterogeneousGraphBuilder']
    
    logger.info("Building heterogeneous graph...")
    
    config = Config()
    embedding_model = EmbeddingModel(config.embedding)
    
    graph = HeterogeneousGraphBuilder(config, embedding_model)
    
    # Filter by years if specified
    year_filter = None
    if args.years:
        year_filter = [int(y) for y in args.years.split(",")]
    
    graph.build_from_data(
        args.data_path,
        year_filter=year_filter,
        compute_embeddings=not args.no_embeddings
    )
    
    # Save graph
    save_dir = args.output or config.paths.graph_dir
    graph.save(save_dir)
    
    # Print statistics
    stats = graph.get_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save embeddings cache
    if not args.no_embeddings:
        cache_path = os.path.join(config.paths.embeddings_dir, "embeddings_cache.json")
        embedding_model.save_cache(cache_path)
    
    logger.info(f"Graph saved to {save_dir}")


def train_link_prediction(args: argparse.Namespace) -> None:
    """Train the link prediction model."""
    _check_dependencies()
    modules = _import_modules()
    
    Config = modules['Config']
    EmbeddingModel = modules['EmbeddingModel']
    HeterogeneousGraphBuilder = modules['HeterogeneousGraphBuilder']
    PlausibilityScorer = modules['PlausibilityScorer']
    
    logger.info("Training link prediction model...")
    
    config = Config()
    
    # Load or build graph
    graph_dir = args.graph_dir or config.paths.graph_dir
    
    embedding_model = EmbeddingModel(config.embedding)
    graph = HeterogeneousGraphBuilder(config, embedding_model)
    
    if os.path.exists(os.path.join(graph_dir, "graph.gpickle")):
        graph.load(graph_dir)
    else:
        logger.info("Graph not found, building from data...")
        graph.build_from_data(args.data_path, compute_embeddings=True)
        graph.save(graph_dir)
    
    # Train link prediction model
    plausibility_scorer = PlausibilityScorer(graph, config.link_prediction)
    plausibility_scorer.train(force_retrain=args.force)
    
    logger.info("Link prediction model training complete")


def evaluate_idea(args: argparse.Namespace) -> None:
    """Evaluate a single research idea."""
    _check_dependencies()
    modules = _import_modules()
    
    Config = modules['Config']
    EmbeddingModel = modules['EmbeddingModel']
    HeterogeneousGraphBuilder = modules['HeterogeneousGraphBuilder']
    InputProcessor = modules['InputProcessor']
    PlausibilityScorer = modules['PlausibilityScorer']
    FinalScorer = modules['FinalScorer']
    
    logger.info("Evaluating research idea...")
    
    config = Config()
    
    # Load graph
    graph_dir = args.graph_dir or config.paths.graph_dir
    
    embedding_model = EmbeddingModel(config.embedding)
    graph = HeterogeneousGraphBuilder(config, embedding_model)
    
    if not os.path.exists(os.path.join(graph_dir, "graph.gpickle")):
        logger.error(f"Graph not found at {graph_dir}. Please run 'build-graph' first.")
        sys.exit(1)
    
    graph.load(graph_dir)
    
    # Load embeddings cache
    cache_path = os.path.join(config.paths.embeddings_dir, "embeddings_cache.json")
    if os.path.exists(cache_path):
        embedding_model.load_cache(cache_path)
    
    # Initialize scorer
    input_processor = InputProcessor(config, embedding_model)
    
    # Load link prediction model if available
    plausibility_scorer = PlausibilityScorer(graph, config.link_prediction)
    model_path = os.path.join(config.link_prediction.model_save_path, "trained_model")
    if os.path.exists(model_path):
        plausibility_scorer.load_model(model_path)
    
    final_scorer = FinalScorer(
        graph, 
        config,
        plausibility_scorer=plausibility_scorer,
        input_processor=input_processor
    )
    
    # Get idea text
    if args.idea:
        idea_text = args.idea
    elif args.idea_file:
        with open(args.idea_file, "r") as f:
            idea_text = f.read()
    elif args.mct_json:
        # Direct MCT JSON input
        mct_dict = json.loads(args.mct_json)
        result = final_scorer.score_mct_dict(mct_dict)
        print(final_scorer.get_score_breakdown(result))
        return
    else:
        logger.error("Please provide an idea using --idea, --idea-file, or --mct-json")
        sys.exit(1)
    
    # Evaluate
    extract_mct = not args.no_extract
    result = final_scorer.score_idea(idea_text, extract_mct=extract_mct)
    
    # Print results
    print(final_scorer.get_score_breakdown(result))
    
    if args.verbose:
        print("\nDetailed Information:")
        print(json.dumps(result.details, indent=2, default=str))


def run_validation(args: argparse.Namespace) -> None:
    """Run the validation pipeline."""
    _check_dependencies()
    modules = _import_modules()
    
    Config = modules['Config']
    ValidationPipeline = modules['ValidationPipeline']
    
    logger.info("Running validation pipeline...")
    
    config = Config()
    
    # Update config from args
    if args.train_years:
        config.validation.train_years = [int(y) for y in args.train_years.split(",")]
    if args.test_year:
        config.validation.test_year = int(args.test_year)
    if args.top_percentile:
        config.validation.top_percentile = float(args.top_percentile)
    
    pipeline = ValidationPipeline(config)
    result = pipeline.run(
        args.data_path,
        train_link_prediction=not args.skip_lp_training
    )
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "ranking_accuracy": result.ranking_accuracy,
                "hit_ratio_at_5": result.hit_ratio_at_5,
                "hit_ratio_at_10": result.hit_ratio_at_10,
                "spearman_correlation": result.spearman_correlation,
                "avg_score_gt": result.avg_score_gt,
                "avg_score_old": result.avg_score_old,
                "avg_score_random": result.avg_score_random,
                "details": result.details,
            }, f, indent=2)
        logger.info(f"Results saved to {args.output}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sci-Idea-EVAL: Research Idea Evaluation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build graph command
    build_parser = subparsers.add_parser(
        "build-graph",
        help="Build heterogeneous MCT graph from data"
    )
    build_parser.add_argument(
        "--data-path", 
        required=True,
        help="Path to JSONL data file"
    )
    build_parser.add_argument(
        "--output",
        help="Output directory for graph (default: cache/graph)"
    )
    build_parser.add_argument(
        "--years",
        help="Comma-separated list of years to include"
    )
    build_parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip computing embeddings"
    )
    
    # Train link prediction command
    train_parser = subparsers.add_parser(
        "train-lp",
        help="Train link prediction model"
    )
    train_parser.add_argument(
        "--data-path",
        required=True,
        help="Path to JSONL data file"
    )
    train_parser.add_argument(
        "--graph-dir",
        help="Directory containing the graph (default: cache/graph)"
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if model exists"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a research idea"
    )
    eval_parser.add_argument(
        "--idea",
        help="Research idea text"
    )
    eval_parser.add_argument(
        "--idea-file",
        help="Path to file containing research idea"
    )
    eval_parser.add_argument(
        "--mct-json",
        help="Direct MCT JSON input"
    )
    eval_parser.add_argument(
        "--graph-dir",
        help="Directory containing the graph"
    )
    eval_parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip MCT extraction (input should be MCT JSON)"
    )
    eval_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information"
    )
    
    # Validation command
    val_parser = subparsers.add_parser(
        "validate",
        help="Run validation pipeline"
    )
    val_parser.add_argument(
        "--data-path",
        required=True,
        help="Path to JSONL data file"
    )
    val_parser.add_argument(
        "--train-years",
        help="Comma-separated years for training (default: 2018,2019,2020,2021)"
    )
    val_parser.add_argument(
        "--test-year",
        type=int,
        help="Year for testing (default: 2022)"
    )
    val_parser.add_argument(
        "--top-percentile",
        type=float,
        help="Top percentile for positive samples (default: 0.1)"
    )
    val_parser.add_argument(
        "--skip-lp-training",
        action="store_true",
        help="Skip link prediction training"
    )
    val_parser.add_argument(
        "--output",
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "build-graph":
        build_graph(args)
    elif args.command == "train-lp":
        train_link_prediction(args)
    elif args.command == "evaluate":
        evaluate_idea(args)
    elif args.command == "validate":
        run_validation(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
