"""
Heterogeneous graph builder module for Sci-Idea-EVAL system.
Constructs the MCT knowledge graph from paper data.
"""

import json
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .config import Config, GraphConfig
from .embedding_model import EmbeddingModel
from .utils import extract_mct_from_paper, get_parent_class, has_valid_mct, load_jsonl

logger = logging.getLogger(__name__)


# Node type constants
NODE_TYPE_PAPER = "paper"
NODE_TYPE_MECHANISM = "mechanism"
NODE_TYPE_CONTEXT = "context"
NODE_TYPE_TASK = "task"

# Edge type constants
EDGE_TYPE_USES = "uses"  # Paper -> Mechanism
EDGE_TYPE_UNDER = "under"  # Paper -> Context
EDGE_TYPE_SOLVES = "solves"  # Paper -> Task
EDGE_TYPE_CITES = "cites"  # Paper -> Paper


@dataclass
class GraphNode:
    """Represents a node in the heterogeneous graph."""
    id: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    """Represents an edge in the heterogeneous graph."""
    source: str
    target: str
    edge_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)


class HeterogeneousGraphBuilder:
    """
    Builder for constructing heterogeneous MCT graphs from paper data.
    
    Graph structure G = (V, E):
    - Node types: Paper (P), Mechanism (M), Context (C), Task (T)
    - Edge types: uses (P->M), under (P->C), solves (P->T), cites (P->P)
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration object.
            embedding_model: Pre-initialized embedding model.
        """
        self.config = config or Config()
        self.embedding_model = embedding_model
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        
        # Track nodes by type
        self.papers: Dict[str, GraphNode] = {}
        self.mechanisms: Dict[str, GraphNode] = {}
        self.contexts: Dict[str, GraphNode] = {}
        self.tasks: Dict[str, GraphNode] = {}
        
        # Parent class mappings
        self.parent_classes: Dict[str, str] = {}
        self.parent_embeddings: Dict[str, np.ndarray] = {}
        
        # Index structures for efficient lookup
        self._paper_to_mechanisms: Dict[str, Set[str]] = defaultdict(set)
        self._paper_to_contexts: Dict[str, Set[str]] = defaultdict(set)
        self._paper_to_tasks: Dict[str, Set[str]] = defaultdict(set)
        self._mechanism_to_papers: Dict[str, Set[str]] = defaultdict(set)
        self._context_to_papers: Dict[str, Set[str]] = defaultdict(set)
        self._task_to_papers: Dict[str, Set[str]] = defaultdict(set)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a node name for consistent matching."""
        return name.strip().lower()
    
    def _get_node_id(self, name: str, node_type: str) -> str:
        """Generate a unique node ID."""
        normalized = self._normalize_name(name)
        return f"{node_type}:{normalized}"
    
    def add_paper(
        self, 
        paper_id: str, 
        year: int, 
        n_citation: int, 
        venue: str,
        title: str = "",
        abstract: str = ""
    ) -> GraphNode:
        """
        Add a paper node to the graph.
        
        Args:
            paper_id: Unique paper identifier.
            year: Publication year.
            n_citation: Number of citations.
            venue: Publication venue.
            title: Paper title.
            abstract: Paper abstract.
            
        Returns:
            The created or existing paper node.
        """
        node_id = f"paper:{paper_id}"
        
        if node_id in self.papers:
            return self.papers[node_id]
        
        node = GraphNode(
            id=node_id,
            node_type=NODE_TYPE_PAPER,
            attributes={
                "paper_id": paper_id,
                "year": year,
                "n_citation": n_citation,
                "venue": venue,
                "title": title,
                "abstract": abstract,
            }
        )
        
        self.papers[node_id] = node
        self.graph.add_node(
            node_id, 
            node_type=NODE_TYPE_PAPER,
            **node.attributes
        )
        
        return node
    
    def add_mechanism(
        self, 
        name: str, 
        embedding: Optional[np.ndarray] = None,
        parent_class: Optional[str] = None
    ) -> GraphNode:
        """
        Add a mechanism node to the graph.
        
        Args:
            name: Mechanism name.
            embedding: Pre-computed embedding vector.
            parent_class: Parent class of the mechanism.
            
        Returns:
            The created or existing mechanism node.
        """
        node_id = self._get_node_id(name, "mechanism")
        
        if node_id in self.mechanisms:
            # Update embedding if provided
            if embedding is not None:
                self.mechanisms[node_id].embedding = embedding
                self.graph.nodes[node_id]["embedding"] = embedding.tolist()
            return self.mechanisms[node_id]
        
        # Determine parent class
        if parent_class is None:
            parent_class = get_parent_class(name)
        
        self.parent_classes[node_id] = parent_class
        
        node = GraphNode(
            id=node_id,
            node_type=NODE_TYPE_MECHANISM,
            attributes={
                "name": name,
                "parent_class": parent_class,
            },
            embedding=embedding
        )
        
        self.mechanisms[node_id] = node
        
        attrs = {"node_type": NODE_TYPE_MECHANISM, "name": name, "parent_class": parent_class}
        if embedding is not None:
            attrs["embedding"] = embedding.tolist()
        
        self.graph.add_node(node_id, **attrs)
        
        return node
    
    def add_context(
        self, 
        name: str, 
        embedding: Optional[np.ndarray] = None
    ) -> GraphNode:
        """
        Add a context node to the graph.
        
        Args:
            name: Context name.
            embedding: Pre-computed embedding vector.
            
        Returns:
            The created or existing context node.
        """
        node_id = self._get_node_id(name, "context")
        
        if node_id in self.contexts:
            if embedding is not None:
                self.contexts[node_id].embedding = embedding
                self.graph.nodes[node_id]["embedding"] = embedding.tolist()
            return self.contexts[node_id]
        
        node = GraphNode(
            id=node_id,
            node_type=NODE_TYPE_CONTEXT,
            attributes={"name": name},
            embedding=embedding
        )
        
        self.contexts[node_id] = node
        
        attrs = {"node_type": NODE_TYPE_CONTEXT, "name": name}
        if embedding is not None:
            attrs["embedding"] = embedding.tolist()
        
        self.graph.add_node(node_id, **attrs)
        
        return node
    
    def add_task(
        self, 
        name: str, 
        embedding: Optional[np.ndarray] = None
    ) -> GraphNode:
        """
        Add a task node to the graph.
        
        Args:
            name: Task name.
            embedding: Pre-computed embedding vector.
            
        Returns:
            The created or existing task node.
        """
        node_id = self._get_node_id(name, "task")
        
        if node_id in self.tasks:
            if embedding is not None:
                self.tasks[node_id].embedding = embedding
                self.graph.nodes[node_id]["embedding"] = embedding.tolist()
            return self.tasks[node_id]
        
        node = GraphNode(
            id=node_id,
            node_type=NODE_TYPE_TASK,
            attributes={"name": name},
            embedding=embedding
        )
        
        self.tasks[node_id] = node
        
        attrs = {"node_type": NODE_TYPE_TASK, "name": name}
        if embedding is not None:
            attrs["embedding"] = embedding.tolist()
        
        self.graph.add_node(node_id, **attrs)
        
        return node
    
    def add_edge(
        self, 
        source_id: str, 
        target_id: str, 
        edge_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an edge to the graph.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of the edge.
            attributes: Optional edge attributes.
        """
        if source_id not in self.graph or target_id not in self.graph:
            logger.warning(
                f"Cannot add edge: node not found. Source: {source_id}, Target: {target_id}"
            )
            return
        
        attrs = {"edge_type": edge_type}
        if attributes:
            attrs.update(attributes)
        
        self.graph.add_edge(source_id, target_id, **attrs)
        
        # Update index structures
        if edge_type == EDGE_TYPE_USES:
            self._paper_to_mechanisms[source_id].add(target_id)
            self._mechanism_to_papers[target_id].add(source_id)
        elif edge_type == EDGE_TYPE_UNDER:
            self._paper_to_contexts[source_id].add(target_id)
            self._context_to_papers[target_id].add(source_id)
        elif edge_type == EDGE_TYPE_SOLVES:
            self._paper_to_tasks[source_id].add(target_id)
            self._task_to_papers[target_id].add(source_id)
    
    def build_from_data(
        self, 
        data_path: str,
        year_filter: Optional[List[int]] = None,
        compute_embeddings: bool = True
    ) -> None:
        """
        Build the graph from JSONL data file.
        
        Args:
            data_path: Path to the JSONL data file.
            year_filter: Optional list of years to include.
            compute_embeddings: Whether to compute embeddings for MCT nodes.
        """
        logger.info(f"Building graph from {data_path}")
        
        # Collect all MCT texts for batch embedding
        all_mechanisms = set()
        all_contexts = set()
        all_tasks = set()
        papers_data = []
        
        # First pass: collect all unique MCT components
        for paper in load_jsonl(data_path):
            if year_filter and paper.get("year") not in year_filter:
                continue
            
            if not has_valid_mct(paper):
                continue
            
            papers_data.append(paper)
            mechanisms, contexts, tasks = extract_mct_from_paper(paper)
            
            all_mechanisms.update(mechanisms)
            all_contexts.update(contexts)
            all_tasks.update(tasks)
        
        logger.info(
            f"Found {len(papers_data)} papers, "
            f"{len(all_mechanisms)} mechanisms, "
            f"{len(all_contexts)} contexts, "
            f"{len(all_tasks)} tasks"
        )
        
        # Compute embeddings if needed
        mechanism_embeddings = {}
        context_embeddings = {}
        task_embeddings = {}
        
        if compute_embeddings and self.embedding_model:
            logger.info("Computing embeddings for MCT components...")
            
            mechanism_list = list(all_mechanisms)
            context_list = list(all_contexts)
            task_list = list(all_tasks)
            
            if mechanism_list:
                embeddings = self.embedding_model.embed_texts(mechanism_list)
                mechanism_embeddings = dict(zip(mechanism_list, embeddings))
            
            if context_list:
                embeddings = self.embedding_model.embed_texts(context_list)
                context_embeddings = dict(zip(context_list, embeddings))
            
            if task_list:
                embeddings = self.embedding_model.embed_texts(task_list)
                task_embeddings = dict(zip(task_list, embeddings))
            
            # Compute parent class embeddings
            parent_classes = set(get_parent_class(m) for m in all_mechanisms)
            if parent_classes:
                parent_list = list(parent_classes)
                parent_embs = self.embedding_model.embed_texts(parent_list)
                self.parent_embeddings = dict(zip(parent_list, parent_embs))
        
        # Second pass: build graph
        for paper in papers_data:
            paper_id = paper["id"]
            
            # Add paper node
            self.add_paper(
                paper_id=paper_id,
                year=paper.get("year", 0),
                n_citation=paper.get("n_citation", 0),
                venue=paper.get("venue", ""),
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
            )
            
            paper_node_id = f"paper:{paper_id}"
            mechanisms, contexts, tasks = extract_mct_from_paper(paper)
            
            # Add mechanism nodes and edges
            for mechanism in mechanisms:
                embedding = mechanism_embeddings.get(mechanism)
                node = self.add_mechanism(mechanism, embedding=embedding)
                self.add_edge(paper_node_id, node.id, EDGE_TYPE_USES)
            
            # Add context nodes and edges
            for context in contexts:
                embedding = context_embeddings.get(context)
                node = self.add_context(context, embedding=embedding)
                self.add_edge(paper_node_id, node.id, EDGE_TYPE_UNDER)
            
            # Add task nodes and edges
            for task in tasks:
                embedding = task_embeddings.get(task)
                node = self.add_task(task, embedding=embedding)
                self.add_edge(paper_node_id, node.id, EDGE_TYPE_SOLVES)
        
        logger.info(
            f"Graph built with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )
    
    def get_papers_with_context_and_task(
        self, 
        context: str, 
        task: str
    ) -> Set[str]:
        """
        Get papers that are connected to both a context and a task.
        
        Args:
            context: Context name.
            task: Task name.
            
        Returns:
            Set of paper node IDs.
        """
        context_id = self._get_node_id(context, "context")
        task_id = self._get_node_id(task, "task")
        
        context_papers = self._context_to_papers.get(context_id, set())
        task_papers = self._task_to_papers.get(task_id, set())
        
        return context_papers & task_papers
    
    def get_mechanisms_for_papers(self, paper_ids: Set[str]) -> Set[str]:
        """
        Get all mechanisms used by a set of papers.
        
        Args:
            paper_ids: Set of paper node IDs.
            
        Returns:
            Set of mechanism node IDs.
        """
        mechanisms = set()
        for paper_id in paper_ids:
            mechanisms.update(self._paper_to_mechanisms.get(paper_id, set()))
        return mechanisms
    
    def get_mechanism_embedding(self, mechanism_id: str) -> Optional[np.ndarray]:
        """Get embedding for a mechanism node."""
        if mechanism_id in self.mechanisms:
            return self.mechanisms[mechanism_id].embedding
        return None
    
    def get_context_embedding(self, context_id: str) -> Optional[np.ndarray]:
        """Get embedding for a context node."""
        if context_id in self.contexts:
            return self.contexts[context_id].embedding
        return None
    
    def get_task_embedding(self, task_id: str) -> Optional[np.ndarray]:
        """Get embedding for a task node."""
        if task_id in self.tasks:
            return self.tasks[task_id].embedding
        return None
    
    def get_parent_embedding(self, mechanism_id: str) -> Optional[np.ndarray]:
        """Get parent class embedding for a mechanism."""
        parent_class = self.parent_classes.get(mechanism_id)
        if parent_class:
            return self.parent_embeddings.get(parent_class)
        return None
    
    def get_task_neighbors(self, task: str) -> List[Dict[str, Any]]:
        """
        Get papers connected to a task with their attributes.
        
        Args:
            task: Task name.
            
        Returns:
            List of paper attribute dictionaries.
        """
        task_id = self._get_node_id(task, "task")
        paper_ids = self._task_to_papers.get(task_id, set())
        
        neighbors = []
        for paper_id in paper_ids:
            if paper_id in self.papers:
                neighbors.append(self.papers[paper_id].attributes)
        
        return neighbors
    
    def get_triples(self) -> List[Tuple[str, str, str]]:
        """
        Get all triples (head, relation, tail) from the graph.
        
        Returns:
            List of (head, relation, tail) tuples.
        """
        triples = []
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            triples.append((u, edge_type, v))
        return triples
    
    def save(self, save_dir: str) -> None:
        """
        Save the graph to disk.
        
        Args:
            save_dir: Directory to save the graph.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save NetworkX graph
        graph_path = os.path.join(save_dir, "graph.gpickle")
        with open(graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        
        # Save node dictionaries
        nodes_path = os.path.join(save_dir, "nodes.pkl")
        with open(nodes_path, "wb") as f:
            pickle.dump({
                "papers": self.papers,
                "mechanisms": self.mechanisms,
                "contexts": self.contexts,
                "tasks": self.tasks,
                "parent_classes": self.parent_classes,
                "parent_embeddings": self.parent_embeddings,
            }, f)
        
        # Save index structures
        index_path = os.path.join(save_dir, "index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump({
                "paper_to_mechanisms": dict(self._paper_to_mechanisms),
                "paper_to_contexts": dict(self._paper_to_contexts),
                "paper_to_tasks": dict(self._paper_to_tasks),
                "mechanism_to_papers": dict(self._mechanism_to_papers),
                "context_to_papers": dict(self._context_to_papers),
                "task_to_papers": dict(self._task_to_papers),
            }, f)
        
        logger.info(f"Graph saved to {save_dir}")
    
    def load(self, save_dir: str) -> None:
        """
        Load the graph from disk.
        
        Args:
            save_dir: Directory containing the saved graph.
        """
        # Load NetworkX graph
        graph_path = os.path.join(save_dir, "graph.gpickle")
        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)
        
        # Load node dictionaries
        nodes_path = os.path.join(save_dir, "nodes.pkl")
        with open(nodes_path, "rb") as f:
            nodes_data = pickle.load(f)
            self.papers = nodes_data["papers"]
            self.mechanisms = nodes_data["mechanisms"]
            self.contexts = nodes_data["contexts"]
            self.tasks = nodes_data["tasks"]
            self.parent_classes = nodes_data["parent_classes"]
            self.parent_embeddings = nodes_data["parent_embeddings"]
        
        # Load index structures
        index_path = os.path.join(save_dir, "index.pkl")
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
            self._paper_to_mechanisms = defaultdict(set, {
                k: set(v) for k, v in index_data["paper_to_mechanisms"].items()
            })
            self._paper_to_contexts = defaultdict(set, {
                k: set(v) for k, v in index_data["paper_to_contexts"].items()
            })
            self._paper_to_tasks = defaultdict(set, {
                k: set(v) for k, v in index_data["paper_to_tasks"].items()
            })
            self._mechanism_to_papers = defaultdict(set, {
                k: set(v) for k, v in index_data["mechanism_to_papers"].items()
            })
            self._context_to_papers = defaultdict(set, {
                k: set(v) for k, v in index_data["context_to_papers"].items()
            })
            self._task_to_papers = defaultdict(set, {
                k: set(v) for k, v in index_data["task_to_papers"].items()
            })
        
        logger.info(f"Graph loaded from {save_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_papers": len(self.papers),
            "num_mechanisms": len(self.mechanisms),
            "num_contexts": len(self.contexts),
            "num_tasks": len(self.tasks),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }
