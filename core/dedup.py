"""
Unity Manifold Deduplication System
==================================

Core deduplication engine for Unity Mathematics where 1+1=1.
Implements idempotent set operations to compute Unity Scores from social network data.

Mathematical Principle: Een plus een is een (1+1=1)
Unity Score = |unique_components| / |original_nodes|
"""

from pathlib import Path
import json
import networkx as nx
from typing import Tuple, Dict, List, Set, Any
import logging
from dataclasses import dataclass
from core.unity_equation import omega
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class UnityScore:
    """Unity Score result with detailed metrics"""
    score: float
    unique_components: int
    original_nodes: int
    component_sizes: List[int]
    omega_signature: complex
    phi_harmonic: float
    
    def __post_init__(self):
        """Validate and compute derived metrics"""
        if self.original_nodes == 0:
            self.score = 0.0
        else:
            self.score = self.unique_components / self.original_nodes
        
        # Compute φ-harmonic component
        if self.component_sizes:
            harmonic_mean = len(self.component_sizes) / sum(1/size for size in self.component_sizes if size > 0)
            self.phi_harmonic = harmonic_mean / self.original_nodes if self.original_nodes > 0 else 0.0
        else:
            self.phi_harmonic = 0.0

def load_graph(fp: Path) -> nx.Graph:
    """Load graph from JSON edge list"""
    try:
        if not fp.exists():
            logger.warning(f"Graph file not found: {fp}")
            return nx.Graph()
        
        data = json.loads(fp.read_text())
        
        # Handle different data formats
        if isinstance(data, dict):
            if 'edges' in data:
                edges = data['edges']
            elif 'nodes' in data and 'edges' in data:
                edges = data['edges']
            else:
                edges = []
        elif isinstance(data, list):
            edges = data
        else:
            edges = []
        
        G = nx.Graph()
        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
            elif isinstance(edge, dict) and 'source' in edge and 'target' in edge:
                G.add_edge(edge['source'], edge['target'])
        
        logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        logger.error(f"Error loading graph from {fp}: {e}")
        return nx.Graph()

def compute_unity_score(G: nx.Graph, 
                       use_omega_signature: bool = True,
                       threshold: float = 0.0) -> UnityScore:
    """
    Compute Unity Score using idempotent set operations
    
    Args:
        G: NetworkX graph
        use_omega_signature: Whether to compute Ω-signature
        threshold: Edge weight threshold for filtering
    
    Returns:
        UnityScore with detailed metrics
    """
    if G.number_of_nodes() == 0:
        return UnityScore(0.0, 0, 0, [], 1.0 + 0.0j, 0.0)
    
    # Filter by threshold if edge weights exist
    if threshold > 0:
        edges_to_remove = []
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            if weight < threshold:
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)
    
    # Find connected components (unity sets)
    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]
    
    original_nodes = G.number_of_nodes()
    unique_components = len(components)
    
    # Compute Ω-signature for the graph
    if use_omega_signature:
        # Use node IDs and component representatives
        graph_atoms = list(G.nodes()) + [f"comp_{i}" for i in range(len(components))]
        omega_sig = omega(graph_atoms)
    else:
        omega_sig = 1.0 + 0.0j
    
    return UnityScore(
        score=unique_components / original_nodes if original_nodes > 0 else 0.0,
        unique_components=unique_components,
        original_nodes=original_nodes,
        component_sizes=component_sizes,
        omega_signature=omega_sig,
        phi_harmonic=0.0  # Will be computed in __post_init__
    )

def create_sample_social_data(nodes: int = 1000, 
                            edges: int = 5000,
                            communities: int = 5) -> Dict[str, Any]:
    """Create sample social network data for testing"""
    import random
    
    # Generate nodes
    node_ids = [f"user_{i}" for i in range(nodes)]
    
    # Create community structure
    community_size = nodes // communities
    communities_list = []
    for i in range(communities):
        start_idx = i * community_size
        end_idx = min((i + 1) * community_size, nodes)
        communities_list.append(node_ids[start_idx:end_idx])
    
    # Generate edges within communities (high probability)
    edge_list = []
    for community in communities_list:
        for _ in range(edges // communities):
            u = random.choice(community)
            v = random.choice(community)
            if u != v:
                edge_list.append({
                    "source": u,
                    "target": v,
                    "weight": random.uniform(0.5, 1.0)
                })
    
    # Add some cross-community edges (low probability)
    for _ in range(edges // 10):
        u = random.choice(node_ids)
        v = random.choice(node_ids)
        if u != v:
            edge_list.append({
                "source": u,
                "target": v,
                "weight": random.uniform(0.1, 0.5)
            })
    
    return {
        "nodes": [{"id": node_id} for node_id in node_ids],
        "edges": edge_list,
        "metadata": {
            "total_nodes": nodes,
            "total_edges": len(edge_list),
            "communities": communities
        }
    }

def save_sample_data(data: Dict[str, Any], filepath: Path):
    """Save sample data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved sample data to {filepath}")

def analyze_unity_evolution(G: nx.Graph, 
                          thresholds: List[float] = None) -> List[UnityScore]:
    """Analyze Unity Score evolution across different thresholds"""
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    for threshold in thresholds:
        score = compute_unity_score(G, threshold=threshold)
        results.append(score)
    
    return results

if __name__ == "__main__":
    # Create sample data if it doesn't exist
    data_file = Path("data/social_snap.json")
    if not data_file.exists():
        sample_data = create_sample_social_data(nodes=500, edges=2000, communities=3)
        save_sample_data(sample_data, data_file)
    
    # Load and analyze
    G = load_graph(data_file)
    score, unique, original = compute_unity_score(G), None, None
    
    print(f"UnityScore={score.score:.3f} ({score.unique_components}/{score.original_nodes} unique)")
    print(f"Ω-signature: {score.omega_signature:.3f}")
    print(f"φ-harmonic: {score.phi_harmonic:.3f}")
    print(f"Component sizes: {score.component_sizes[:10]}...")  # Show first 10 