"""
Plausibility scorer module for Sci-Idea-EVAL system.
Computes plausibility score using link prediction models (RotatE/ComplEx).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config, LinkPredictionConfig
from .graph_builder import (
    EDGE_TYPE_SOLVES,
    EDGE_TYPE_UNDER,
    EDGE_TYPE_USES,
    HeterogeneousGraphBuilder,
)
from .input_processor import MCTTriple

logger = logging.getLogger(__name__)


class PlausibilityScorer:
    """
    Computes plausibility score using knowledge graph link prediction.
    
    Uses PyKEEN to train RotatE or ComplEx models on the MCT graph
    and predicts the likelihood of new mechanism-context-task combinations.
    
    Score: P(link | m_new, (c_new, t_new))
    """
    
    def __init__(
        self, 
        graph: HeterogeneousGraphBuilder,
        config: Optional[LinkPredictionConfig] = None
    ):
        """
        Initialize the plausibility scorer.
        
        Args:
            graph: The heterogeneous MCT graph.
            config: Link prediction configuration.
        """
        self.graph = graph
        self.config = config or LinkPredictionConfig()
        self.model = None
        self.training_triples = None
        self.entity_to_id: Dict[str, int] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.id_to_relation: Dict[int, str] = {}
        
    def _prepare_triples(self) -> List[Tuple[str, str, str]]:
        """
        Prepare triples from the graph for PyKEEN training.
        
        Returns:
            List of (head, relation, tail) tuples.
        """
        triples = self.graph.get_triples()
        
        # Build entity and relation mappings
        entities = set()
        relations = set()
        
        for head, relation, tail in triples:
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
        
        self.entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}
        
        logger.info(f"Prepared {len(triples)} triples, {len(entities)} entities, {len(relations)} relations")
        
        return triples
    
    def train(self, force_retrain: bool = False) -> None:
        """
        Train the link prediction model using PyKEEN.
        
        Args:
            force_retrain: If True, retrain even if a saved model exists.
        """
        try:
            from pykeen.pipeline import pipeline
            from pykeen.triples import TriplesFactory
        except ImportError:
            logger.error("PyKEEN is not installed. Please install it with: pip install pykeen")
            raise
        
        model_path = os.path.join(self.config.model_save_path, "trained_model")
        
        # Check for existing model
        if not force_retrain and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            self.load_model(model_path)
            return
        
        # Prepare training data
        triples = self._prepare_triples()
        
        if not triples:
            logger.warning("No triples available for training")
            return
        
        # Convert to numpy array
        triple_array = np.array([
            [self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]]
            for h, r, t in triples
        ])
        
        # Create TriplesFactory
        tf = TriplesFactory.from_labeled_triples(
            np.array([[h, r, t] for h, r, t in triples]),
            create_inverse_triples=True,
        )
        
        # Update mappings from TriplesFactory
        self.entity_to_id = tf.entity_to_id
        self.relation_to_id = tf.relation_to_id
        self.id_to_entity = {v: k for k, v in tf.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in tf.relation_to_id.items()}
        
        logger.info(f"Training {self.config.model_name} model...")
        
        # Run training pipeline
        result = pipeline(
            training=tf,
            model=self.config.model_name,
            model_kwargs=dict(
                embedding_dim=self.config.embedding_dim,
            ),
            training_kwargs=dict(
                num_epochs=self.config.num_epochs,
                batch_size=self.config.batch_size,
            ),
            optimizer_kwargs=dict(
                lr=self.config.learning_rate,
            ),
            negative_sampler_kwargs=dict(
                num_negs_per_pos=self.config.num_negatives,
            ),
            random_seed=42,
        )
        
        self.model = result.model
        self.training_triples = tf
        
        # Save model
        os.makedirs(self.config.model_save_path, exist_ok=True)
        result.save_to_directory(model_path)
        
        # Save mappings
        import pickle
        mappings_path = os.path.join(self.config.model_save_path, "mappings.pkl")
        with open(mappings_path, "wb") as f:
            pickle.dump({
                "entity_to_id": self.entity_to_id,
                "relation_to_id": self.relation_to_id,
            }, f)
        
        logger.info(f"Model trained and saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model directory.
        """
        try:
            from pykeen.pipeline import pipeline_from_path
        except ImportError:
            logger.error("PyKEEN is not installed")
            raise
        
        import pickle
        
        # Load model
        result = pipeline_from_path(model_path)
        self.model = result.model
        self.training_triples = result.training
        
        # Load mappings
        mappings_path = os.path.join(os.path.dirname(model_path), "mappings.pkl")
        if os.path.exists(mappings_path):
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
                self.entity_to_id = mappings["entity_to_id"]
                self.relation_to_id = mappings["relation_to_id"]
        else:
            # Use mappings from TriplesFactory
            self.entity_to_id = self.training_triples.entity_to_id
            self.relation_to_id = self.training_triples.relation_to_id
        
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        
        logger.info(f"Model loaded from {model_path}")
    
    def _predict_link_score(
        self, 
        head: str, 
        relation: str, 
        tail: str
    ) -> float:
        """
        Predict the score for a single triple.
        
        Args:
            head: Head entity ID.
            relation: Relation type.
            tail: Tail entity ID.
            
        Returns:
            Predicted score (probability-like value in [0, 1]).
        """
        if self.model is None:
            logger.warning("Model not trained. Call train() first.")
            return 0.5
        
        # Check if entities exist
        if head not in self.entity_to_id:
            logger.debug(f"Head entity not in graph: {head}")
            return 0.3  # Low score for unknown entities
        
        if tail not in self.entity_to_id:
            logger.debug(f"Tail entity not in graph: {tail}")
            return 0.3
        
        if relation not in self.relation_to_id:
            logger.debug(f"Relation not in graph: {relation}")
            return 0.3
        
        try:
            # Create triple tensor
            h_id = self.entity_to_id[head]
            r_id = self.relation_to_id[relation]
            t_id = self.entity_to_id[tail]
            
            triple = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long)
            
            if hasattr(self.model, 'device'):
                triple = triple.to(self.model.device)
            
            # Get score
            with torch.no_grad():
                score = self.model.score_hrt(triple)
                # Convert to probability using sigmoid
                prob = torch.sigmoid(score).item()
            
            return float(prob)
        except Exception as e:
            logger.warning(f"Error predicting link score: {e}")
            return 0.3
    
    def _find_similar_entity(
        self, 
        name: str, 
        entity_type: str
    ) -> Optional[str]:
        """
        Find a similar entity in the graph based on name matching.
        
        Args:
            name: Entity name to search for.
            entity_type: Type of entity ("mechanism", "context", "task").
            
        Returns:
            Matching entity ID if found, None otherwise.
        """
        normalized = name.strip().lower()
        target_id = f"{entity_type}:{normalized}"
        
        if target_id in self.entity_to_id:
            return target_id
        
        # Try partial matching
        for entity in self.entity_to_id.keys():
            if entity.startswith(f"{entity_type}:"):
                entity_name = entity.split(":", 1)[1]
                if normalized in entity_name or entity_name in normalized:
                    return entity
        
        return None
    
    def score_single_combination(
        self,
        mechanism: str,
        context: str,
        task: str
    ) -> Tuple[float, Dict]:
        """
        Score plausibility for a single M-C-T combination.
        
        Args:
            mechanism: Mechanism name.
            context: Context name.
            task: Task name.
            
        Returns:
            Tuple of (plausibility_score, details_dict).
        """
        # Get entity IDs
        mech_id = self._find_similar_entity(mechanism, "mechanism")
        context_id = self._find_similar_entity(context, "context")
        task_id = self._find_similar_entity(task, "task")
        
        details = {
            "mechanism": mechanism,
            "context": context,
            "task": task,
            "mechanism_found": mech_id is not None,
            "context_found": context_id is not None,
            "task_found": task_id is not None,
        }
        
        # We need a paper node to form the triples
        # For new ideas, we create virtual triples
        # Score = average of (paper->uses->mechanism, paper->under->context, paper->solves->task)
        
        # For evaluation, we use the mechanism as the central node
        # and check its compatibility with context and task
        
        scores = []
        
        if mech_id and context_id:
            # Check if mechanism is compatible with context
            # Use any paper that uses this mechanism
            for paper_id in self.graph._mechanism_to_papers.get(mech_id, []):
                score = self._predict_link_score(paper_id, EDGE_TYPE_UNDER, context_id)
                scores.append(score)
                if len(scores) >= 5:  # Limit samples
                    break
        
        if mech_id and task_id:
            # Check if mechanism is compatible with task
            for paper_id in self.graph._mechanism_to_papers.get(mech_id, []):
                score = self._predict_link_score(paper_id, EDGE_TYPE_SOLVES, task_id)
                scores.append(score)
                if len(scores) >= 5:
                    break
        
        if not scores:
            # Fallback: check direct compatibility using embeddings
            details["fallback"] = True
            return 0.5, details
        
        avg_score = float(np.mean(scores))
        details["num_samples"] = len(scores)
        details["individual_scores"] = scores
        
        return avg_score, details
    
    def score(self, mct_triple: MCTTriple) -> Tuple[float, Dict]:
        """
        Score plausibility for an MCT triple.
        
        Args:
            mct_triple: The extracted MCT triple.
            
        Returns:
            Tuple of (plausibility_score, details_dict).
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
            for context in mct_triple.contexts:
                for task in mct_triple.tasks:
                    score, details = self.score_single_combination(
                        mechanism, context, task
                    )
                    scores.append(score)
                    all_details.append(details)
        
        if not scores:
            return 0.0, {"error": "no_valid_combinations"}
        
        avg_score = float(np.mean(scores))
        
        return avg_score, {
            "avg_plausibility": avg_score,
            "num_combinations": len(scores),
            "combination_details": all_details,
        }
    
    def is_plausible(
        self, 
        mct_triple: MCTTriple, 
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if an MCT triple passes the plausibility threshold.
        
        Args:
            mct_triple: The extracted MCT triple.
            threshold: Plausibility threshold (uses config default if not provided).
            
        Returns:
            True if plausibility score exceeds threshold.
        """
        if threshold is None:
            threshold = self.config.threshold if hasattr(self.config, 'threshold') else 0.3
        
        score, _ = self.score(mct_triple)
        return score > threshold
