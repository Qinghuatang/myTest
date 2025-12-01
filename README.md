# Sci-Idea-EVAL (MCT-Edition)

A benchmark system for automatically evaluating the quality of LLM-generated research ideas based on **Novelty**, **Plausibility**, and **Utility** scores.

## Overview

Sci-Idea-EVAL uses a heterogeneous knowledge graph built from NLP research papers (2018-2022) to evaluate new research ideas. The system analyzes **Mechanism-Context-Task (MCT)** triples extracted from ideas and computes:

- **Novelty Score (S_Nov)**: Measures the semantic jump distance of the proposed mechanism from historically used methods
- **Plausibility Score (S_Plaus)**: Evaluates structural compatibility using knowledge graph link prediction
- **Utility Score (S_Util)**: Combines task market heat with mechanism-context demand matching

**Final Score Formula:**
```
Final_Score = S_Nov × S_Util × I(S_Plaus > θ)
```
where I(·) is an indicator function and θ is the plausibility threshold (default: 0.3).

## Installation

```bash
# Clone the repository
git clone https://github.com/Qinghuatang/myTest.git
cd myTest

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Ollama with llama3.1:70b (for MCT extraction from free-text ideas)

## Quick Start

### 1. Build the Knowledge Graph

```bash
python -m sci_idea_eval.main build-graph --data-path mct_results_v2.jsonl
```

This creates a heterogeneous graph with:
- Paper nodes (with year, citations, venue attributes)
- Mechanism nodes (with embeddings and parent classes)
- Context nodes (with embeddings)
- Task nodes (with embeddings)

### 2. Train Link Prediction Model

```bash
python -m sci_idea_eval.main train-lp --data-path mct_results_v2.jsonl
```

Trains a RotatE model for plausibility scoring.

### 3. Evaluate a Research Idea

```bash
# Evaluate from text (requires Ollama)
python -m sci_idea_eval.main evaluate --idea "We propose using contrastive learning for low-resource machine translation by leveraging multilingual embeddings"

# Evaluate from MCT JSON directly
python -m sci_idea_eval.main evaluate --mct-json '{"mechanism": ["Contrastive Learning"], "context": ["Low-resource Setting"], "task": ["Machine Translation"]}'
```

### 4. Run Validation Experiments

```bash
python -m sci_idea_eval.main validate --data-path mct_results_v2.jsonl
```

Performs retrospective validation using 2018-2021 papers for training and 2022 top papers for testing.

## Architecture

```
sci_idea_eval/
├── __init__.py              # Package initialization
├── config.py                # Configuration settings
├── graph_builder.py         # Heterogeneous graph construction
├── input_processor.py       # MCT extraction (Ollama) and vectorization
├── embedding_model.py       # SciBERT/Specter embeddings
├── novelty_scorer.py        # Semantic novelty evaluation
├── plausibility_scorer.py   # Link prediction (PyKEEN)
├── utility_scorer.py        # Task heat and demand matching
├── final_scorer.py          # Gated product scoring
├── validation_pipeline.py   # Retrospective validation
├── utils.py                 # Utility functions
└── main.py                  # CLI entry point
```

## Scoring Modules

### Novelty Scorer

Computes semantic jump distance from historical mechanisms:

```
S_Nov = α × D_min + (1-α) × D_parent
```

- `D_min`: Minimum cosine distance to historical mechanisms
- `D_parent`: Minimum distance between parent classes
- `α`: Weight parameter (default: 0.7)

Includes isolation penalty for mechanisms with very low similarity to all known mechanisms.

### Plausibility Scorer

Uses PyKEEN's RotatE model trained on the MCT graph:

```
S_Plaus = P(link | m_new, (c_new, t_new))
```

### Utility Scorer

Combines task popularity with mechanism-context fit:

```
S_Util = Normalize(Heat(t)) × Fit(m, c)
```

Where:
- `Heat(t)`: Citation-weighted activity in the task area
- `Fit(m, c)`: Cosine similarity between mechanism and context embeddings

## Configuration

Key configuration options in `config.py`:

```python
# Ollama settings
ollama.base_url = "http://localhost:11434"
ollama.model = "llama3.1:70b"

# Scoring parameters
novelty.alpha = 0.7
plausibility.threshold = 0.3

# Link prediction
link_prediction.model_name = "RotatE"
link_prediction.num_epochs = 100
```

## Data Format

The input JSONL file should contain papers with the following structure:

```json
{
  "id": "paper_id",
  "year": 2022,
  "title": "Paper Title",
  "n_citation": 50,
  "venue": "ACL",
  "extracted_mct": {
    "mechanism": ["Method1", "Method2"],
    "context": ["Context1"],
    "task": ["Task1"]
  },
  "abstract": "Paper abstract..."
}
```

## Validation Metrics

The validation pipeline computes:

1. **Ranking Accuracy**: Percentage of triplets where Score(GT) > Score(Old) > Score(Random)
2. **Hit Ratio @ K**: Proportion of ground truth papers in top-K scored items
3. **Spearman Correlation**: Correlation between scores and actual citation counts

## API Usage

```python
from sci_idea_eval import Config, FinalScorer, HeterogeneousGraphBuilder, EmbeddingModel

# Initialize
config = Config()
embedding_model = EmbeddingModel(config.embedding)
graph = HeterogeneousGraphBuilder(config, embedding_model)
graph.load("cache/graph")

scorer = FinalScorer(graph, config)

# Evaluate an MCT triple
mct = {"mechanism": ["Transformer"], "context": ["NLP"], "task": ["Translation"]}
result = scorer.score_mct_dict(mct)

print(f"Final Score: {result.final_score}")
print(f"Novelty: {result.novelty_score}")
print(f"Plausibility: {result.plausibility_score}")
print(f"Utility: {result.utility_score}")
```

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{sci_idea_eval,
  title = {Sci-Idea-EVAL: A Benchmark for Research Idea Evaluation},
  year = {2024},
  url = {https://github.com/Qinghuatang/myTest}
}
```