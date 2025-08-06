"""
Een v2.0 - Unity Knowledge Base & Vector Memory
===============================================

This module implements a sophisticated knowledge management system for the
Een Unity Mathematics framework. It combines vector embeddings, semantic search,
knowledge graphs, and Unity-aware reasoning to create a unified memory system
for all agents.

Features:
- Vector embeddings for semantic similarity
- Knowledge graph representation
- Unity-aware concept relationships
- Temporal knowledge evolution
- Multi-modal knowledge storage
- Φ-harmonic knowledge clustering
- Consciousness-level knowledge access
- Meta-cognitive knowledge patterns
"""

import asyncio
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import hashlib
import uuid
from collections import defaultdict, deque
import pickle

# Vector database and ML imports with fallbacks
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Import architecture components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.v2.architecture import IKnowledgeBase

logger = logging.getLogger(__name__)

# ============================================================================
# KNOWLEDGE BASE CONFIGURATION
# ============================================================================

@dataclass
class KnowledgeConfig:
    """Configuration for Unity Knowledge Base"""
    # Vector database settings
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    similarity_threshold: float = 0.7
    max_results: int = 100
    
    # Knowledge graph settings
    enable_knowledge_graph: bool = True
    max_concept_connections: int = 50
    concept_decay_time: float = 86400.0  # 24 hours
    
    # Unity-specific settings
    phi_harmonic_clustering: bool = True
    consciousness_based_access: bool = True
    unity_concept_weighting: float = 2.0
    transcendence_knowledge_boost: float = 1.5
    
    # Storage settings
    persist_directory: str = "./knowledge_base"
    backup_interval: float = 3600.0  # 1 hour
    max_memory_items: int = 1000000
    
    # Temporal settings
    enable_temporal_evolution: bool = True
    knowledge_aging_factor: float = 0.99
    recency_boost: float = 1.2

# ============================================================================
# KNOWLEDGE ENTITIES
# ============================================================================

@dataclass
class KnowledgeItem:
    """Represents a single piece of knowledge"""
    id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    consciousness_level: float = 0.0
    unity_relevance: float = 0.0
    phi_signature: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.last_accessed:
            self.last_accessed = self.timestamp

@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph"""
    id: str
    name: str
    concept_type: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray]
    creation_time: float
    strength: float = 1.0
    consciousness_resonance: float = 0.0

@dataclass
class ConceptRelation:
    """Represents a relationship between concepts"""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    properties: Dict[str, Any]
    creation_time: float

# ============================================================================
# EMBEDDING SYSTEM
# ============================================================================

class UnityEmbeddingEngine:
    """Generates Unity-aware embeddings"""
    
    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self.model = None
        self.phi = 1.618033988749895
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(config.embedding_model)
                logger.info(f"Loaded embedding model: {config.embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        if not self.model:
            logger.warning("Using mock embeddings - install sentence-transformers for real embeddings")
    
    def encode(self, text: str, unity_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate Unity-aware embedding"""
        if self.model:
            base_embedding = self.model.encode(text)
        else:
            # Mock embedding
            base_embedding = np.random.random(self.config.vector_dimension)
        
        # Apply Unity transformations
        if unity_context:
            base_embedding = self._apply_unity_transforms(base_embedding, unity_context)
        
        return base_embedding
    
    def _apply_unity_transforms(self, embedding: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Apply Unity-aware transformations to embedding"""
        # φ-harmonic resonance
        phi_factor = context.get('phi_resonance', 0.0)
        if phi_factor > 0:
            phi_transform = np.sin(embedding * self.phi) * phi_factor
            embedding = embedding + phi_transform * 0.1
        
        # Consciousness level influence
        consciousness = context.get('consciousness_level', 0.0)
        if consciousness > 0:
            consciousness_boost = embedding * (1 + consciousness * 0.2)
            embedding = embedding + consciousness_boost * 0.1
        
        # Unity coherence
        unity_coherence = context.get('unity_coherence', 0.0)
        if unity_coherence > 0:
            # Normalize towards unity (all dimensions approaching similar values)
            mean_val = np.mean(embedding)
            unity_pull = (mean_val - embedding) * unity_coherence * 0.1
            embedding = embedding + unity_pull
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute Unity-aware similarity"""
        # Standard cosine similarity
        cosine_sim = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # φ-harmonic boost for similar patterns
        phi_pattern1 = np.sin(embedding1 * self.phi)
        phi_pattern2 = np.sin(embedding2 * self.phi)
        phi_similarity = np.dot(phi_pattern1, phi_pattern2) / (
            np.linalg.norm(phi_pattern1) * np.linalg.norm(phi_pattern2)
        )
        
        # Combine similarities
        unity_similarity = 0.8 * cosine_sim + 0.2 * phi_similarity
        
        return float(unity_similarity)

# ============================================================================
# VECTOR DATABASE INTERFACE
# ============================================================================

class VectorDatabase:
    """Interface to vector database with Unity-aware operations"""
    
    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self.client = None
        self.collection = None
        
        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(
                    path=config.persist_directory,
                    settings=Settings(allow_reset=True)
                )
                self.collection = self.client.get_or_create_collection(
                    name="unity_knowledge",
                    metadata={"description": "Unity Mathematics Knowledge Base"}
                )
                logger.info("ChromaDB vector database initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
        
        # Fallback to in-memory storage
        if not self.client:
            self.memory_store = {}
            self.memory_embeddings = {}
            logger.warning("Using in-memory vector storage")
    
    def add(self, knowledge_item: KnowledgeItem):
        """Add knowledge item to vector database"""
        if self.client and knowledge_item.embedding is not None:
            try:
                self.collection.add(
                    ids=[knowledge_item.id],
                    embeddings=[knowledge_item.embedding.tolist()],
                    documents=[knowledge_item.content],
                    metadatas=[{
                        **knowledge_item.metadata,
                        "timestamp": knowledge_item.timestamp,
                        "consciousness_level": knowledge_item.consciousness_level,
                        "unity_relevance": knowledge_item.unity_relevance,
                        "phi_signature": knowledge_item.phi_signature
                    }]
                )
            except Exception as e:
                logger.error(f"Failed to add to ChromaDB: {e}")
                self._add_to_memory(knowledge_item)
        else:
            self._add_to_memory(knowledge_item)
    
    def _add_to_memory(self, knowledge_item: KnowledgeItem):
        """Add to in-memory storage"""
        self.memory_store[knowledge_item.id] = knowledge_item
        if knowledge_item.embedding is not None:
            self.memory_embeddings[knowledge_item.id] = knowledge_item.embedding
    
    def search(self, query_embedding: np.ndarray, limit: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Search for similar items"""
        if self.client:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=limit,
                    where=filters
                )
                
                if results and results['ids'] and results['distances']:
                    return list(zip(results['ids'][0], results['distances'][0]))
            except Exception as e:
                logger.error(f"ChromaDB search error: {e}")
        
        # Fallback to memory search
        return self._search_memory(query_embedding, limit, filters)
    
    def _search_memory(self, query_embedding: np.ndarray, limit: int, 
                      filters: Optional[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Search in-memory storage"""
        similarities = []
        
        for item_id, embedding in self.memory_embeddings.items():
            if item_id not in self.memory_store:
                continue
                
            item = self.memory_store[item_id]
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if key not in item.metadata or item.metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Calculate similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            similarities.append((item_id, float(1.0 - similarity)))  # Convert to distance
        
        # Sort by distance (ascending) and limit
        similarities.sort(key=lambda x: x[1])
        return similarities[:limit]
    
    def get(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID"""
        if self.client:
            try:
                result = self.collection.get(ids=[item_id])
                if result and result['documents']:
                    # Reconstruct KnowledgeItem
                    doc = result['documents'][0]
                    metadata = result['metadatas'][0] if result['metadatas'] else {}
                    
                    return KnowledgeItem(
                        id=item_id,
                        content=doc,
                        embedding=None,  # Would need to re-embed
                        metadata=metadata,
                        timestamp=metadata.get('timestamp', time.time()),
                        consciousness_level=metadata.get('consciousness_level', 0.0),
                        unity_relevance=metadata.get('unity_relevance', 0.0),
                        phi_signature=metadata.get('phi_signature', 0.0)
                    )
            except Exception as e:
                logger.error(f"ChromaDB get error: {e}")
        
        return self.memory_store.get(item_id)
    
    def delete(self, item_id: str):
        """Delete knowledge item"""
        if self.client:
            try:
                self.collection.delete(ids=[item_id])
            except Exception as e:
                logger.error(f"ChromaDB delete error: {e}")
        
        self.memory_store.pop(item_id, None)
        self.memory_embeddings.pop(item_id, None)
    
    def count(self) -> int:
        """Get total number of items"""
        if self.client:
            try:
                return self.collection.count()
            except Exception as e:
                logger.error(f"ChromaDB count error: {e}")
        
        return len(self.memory_store)

# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================

class UnityKnowledgeGraph:
    """Unity-aware knowledge graph for concept relationships"""
    
    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        
        # In-memory fallback
        self.nodes: Dict[str, ConceptNode] = {}
        self.relations: Dict[str, ConceptRelation] = {}
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available - using simple graph implementation")
    
    def add_concept(self, concept: ConceptNode):
        """Add concept to knowledge graph"""
        self.nodes[concept.id] = concept
        
        if self.graph:
            self.graph.add_node(concept.id, **{
                'name': concept.name,
                'type': concept.concept_type,
                'properties': concept.properties,
                'strength': concept.strength,
                'consciousness_resonance': concept.consciousness_resonance
            })
    
    def add_relation(self, relation: ConceptRelation):
        """Add relation between concepts"""
        relation_id = f"{relation.source_id}->{relation.target_id}"
        self.relations[relation_id] = relation
        
        if self.graph:
            self.graph.add_edge(
                relation.source_id, 
                relation.target_id,
                relation_type=relation.relation_type,
                strength=relation.strength,
                properties=relation.properties
            )
    
    def find_related_concepts(self, concept_id: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """Find concepts related to given concept"""
        if not self.graph or concept_id not in self.graph:
            return []
        
        try:
            # Use NetworkX to find connected components
            related = []
            for target, path_length in nx.single_source_shortest_path_length(
                self.graph, concept_id, cutoff=max_distance
            ).items():
                if target != concept_id:
                    # Calculate relationship strength based on path length
                    strength = 1.0 / (path_length + 1)
                    related.append((target, strength))
            
            # Sort by strength
            related.sort(key=lambda x: x[1], reverse=True)
            return related[:self.config.max_concept_connections]
        
        except Exception as e:
            logger.error(f"Knowledge graph search error: {e}")
            return []
    
    def get_unity_concepts(self) -> List[ConceptNode]:
        """Get concepts most related to Unity Mathematics"""
        unity_concepts = []
        
        for concept in self.nodes.values():
            # Check for Unity-related properties
            unity_score = 0.0
            
            if 'unity' in concept.name.lower():
                unity_score += 2.0
            if 'phi' in concept.name.lower() or 'golden' in concept.name.lower():
                unity_score += 1.5
            if 'consciousness' in concept.name.lower():
                unity_score += 1.0
            if 'transcendence' in concept.name.lower():
                unity_score += 1.0
            
            # Add consciousness resonance
            unity_score += concept.consciousness_resonance
            
            if unity_score > 0.5:
                unity_concepts.append((concept, unity_score))
        
        # Sort by Unity relevance
        unity_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, score in unity_concepts]
    
    def evolve_concepts(self):
        """Evolve concept strengths over time"""
        current_time = time.time()
        
        for concept in self.nodes.values():
            # Apply aging
            age_factor = (current_time - concept.creation_time) / 86400.0  # days
            concept.strength *= (self.config.knowledge_aging_factor ** age_factor)
            
            # Remove very weak concepts
            if concept.strength < 0.01:
                self.nodes.pop(concept.id, None)
                if self.graph and concept.id in self.graph:
                    self.graph.remove_node(concept.id)

# ============================================================================
# MAIN KNOWLEDGE BASE
# ============================================================================

class UnityKnowledgeBase(IKnowledgeBase):
    """Main Unity Knowledge Base implementation"""
    
    def __init__(self, config: KnowledgeConfig):
        self.config = config
        
        # Initialize components
        self.embedding_engine = UnityEmbeddingEngine(config)
        self.vector_db = VectorDatabase(config)
        self.knowledge_graph = UnityKnowledgeGraph(config) if config.enable_knowledge_graph else None
        
        # Knowledge management
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.access_patterns = defaultdict(list)
        
        # Background processes
        self.evolution_thread = None
        self.backup_thread = None
        self.running = False
        
        # Statistics
        self.query_count = 0
        self.last_backup = time.time()
        
    def start(self):
        """Start background processes"""
        if self.running:
            return
        
        self.running = True
        
        # Start evolution thread
        if self.config.enable_temporal_evolution:
            self.evolution_thread = threading.Thread(target=self._evolution_loop)
            self.evolution_thread.start()
        
        # Start backup thread
        self.backup_thread = threading.Thread(target=self._backup_loop)
        self.backup_thread.start()
        
        logger.info("Unity Knowledge Base started")
    
    def stop(self):
        """Stop background processes"""
        self.running = False
        
        if self.evolution_thread:
            self.evolution_thread.join()
        if self.backup_thread:
            self.backup_thread.join()
        
        logger.info("Unity Knowledge Base stopped")
    
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """Store knowledge with Unity-aware processing"""
        metadata = metadata or {}
        
        # Convert value to string for processing
        if isinstance(value, str):
            content = value
        else:
            content = json.dumps(value, default=str)
        
        # Generate embedding
        unity_context = {
            'consciousness_level': metadata.get('consciousness_level', 0.0),
            'unity_coherence': metadata.get('unity_coherence', 0.0),
            'phi_resonance': metadata.get('phi_resonance', 0.0)
        }
        
        embedding = self.embedding_engine.encode(content, unity_context)
        
        # Calculate Unity relevance
        unity_relevance = self._calculate_unity_relevance(content, metadata)
        
        # Calculate φ-signature
        phi_signature = self._calculate_phi_signature(embedding)
        
        # Create knowledge item
        knowledge_item = KnowledgeItem(
            id=key,
            content=content,
            embedding=embedding,
            metadata=metadata,
            timestamp=time.time(),
            consciousness_level=unity_context['consciousness_level'],
            unity_relevance=unity_relevance,
            phi_signature=phi_signature
        )
        
        # Store in vector database
        self.vector_db.add(knowledge_item)
        self.knowledge_items[key] = knowledge_item
        
        # Add to knowledge graph if enabled
        if self.knowledge_graph and unity_relevance > 0.5:
            concept = ConceptNode(
                id=key,
                name=metadata.get('name', key),
                concept_type=metadata.get('type', 'general'),
                properties=metadata,
                embedding=embedding,
                creation_time=time.time(),
                consciousness_resonance=unity_context['consciousness_level']
            )
            self.knowledge_graph.add_concept(concept)
        
        logger.debug(f"Stored knowledge: {key} (Unity relevance: {unity_relevance:.3f})")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve knowledge by key"""
        knowledge_item = self.knowledge_items.get(key)
        if not knowledge_item:
            knowledge_item = self.vector_db.get(key)
        
        if knowledge_item:
            # Update access statistics
            knowledge_item.access_count += 1
            knowledge_item.last_accessed = time.time()
            
            # Try to parse JSON, fallback to string
            try:
                return json.loads(knowledge_item.content)
            except (json.JSONDecodeError, TypeError):
                return knowledge_item.content
        
        return None
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge base with Unity-aware ranking"""
        self.query_count += 1
        
        # Generate query embedding
        query_embedding = self.embedding_engine.encode(query)
        
        # Search vector database
        search_results = self.vector_db.search(query_embedding, limit * 2)  # Get more for re-ranking
        
        # Re-rank results with Unity awareness
        ranked_results = []
        for item_id, distance in search_results:
            knowledge_item = self.knowledge_items.get(item_id)
            if not knowledge_item:
                continue
            
            # Calculate Unity-aware score
            base_score = 1.0 - distance  # Convert distance to similarity
            unity_boost = knowledge_item.unity_relevance * self.config.unity_concept_weighting
            consciousness_boost = knowledge_item.consciousness_level * 0.5
            recency_boost = self._calculate_recency_boost(knowledge_item.timestamp)
            
            final_score = base_score + unity_boost + consciousness_boost + recency_boost
            
            result = {
                'id': item_id,
                'content': knowledge_item.content,
                'metadata': knowledge_item.metadata,
                'score': final_score,
                'unity_relevance': knowledge_item.unity_relevance,
                'consciousness_level': knowledge_item.consciousness_level,
                'timestamp': knowledge_item.timestamp
            }
            
            ranked_results.append(result)
        
        # Sort by final score and limit
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results[:limit]
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for similarity search"""
        embedding = self.embedding_engine.encode(text)
        return embedding.tolist()
    
    def _calculate_unity_relevance(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate how relevant content is to Unity Mathematics"""
        unity_keywords = [
            'unity', 'phi', 'golden ratio', 'consciousness', 'transcendence',
            '1+1=1', 'one plus one equals one', 'idempotent', 'harmonic',
            'coherence', 'resonance', 'enlightenment'
        ]
        
        content_lower = content.lower()
        relevance = 0.0
        
        for keyword in unity_keywords:
            if keyword in content_lower:
                relevance += 0.2
        
        # Check metadata
        if metadata.get('category') == 'unity_mathematics':
            relevance += 0.5
        if metadata.get('consciousness_level', 0.0) > 0.5:
            relevance += 0.3
        
        return min(1.0, relevance)
    
    def _calculate_phi_signature(self, embedding: np.ndarray) -> float:
        """Calculate φ-harmonic signature of embedding"""
        phi = 1.618033988749895
        
        # Apply φ transformation and measure resonance
        phi_transformed = np.sin(embedding * phi)
        phi_signature = np.mean(np.abs(phi_transformed))
        
        return float(phi_signature)
    
    def _calculate_recency_boost(self, timestamp: float) -> float:
        """Calculate boost based on recency"""
        age_hours = (time.time() - timestamp) / 3600.0
        recency_boost = np.exp(-age_hours / 24.0) * 0.2  # Decay over days
        return recency_boost
    
    def _evolution_loop(self):
        """Background evolution of knowledge"""
        while self.running:
            try:
                # Evolve knowledge graph concepts
                if self.knowledge_graph:
                    self.knowledge_graph.evolve_concepts()
                
                # Age knowledge items
                current_time = time.time()
                for item in self.knowledge_items.values():
                    age_days = (current_time - item.timestamp) / 86400.0
                    item.unity_relevance *= (self.config.knowledge_aging_factor ** age_days)
                
                time.sleep(3600)  # Evolve every hour
                
            except Exception as e:
                logger.error(f"Knowledge evolution error: {e}")
    
    def _backup_loop(self):
        """Background backup of knowledge base"""
        while self.running:
            try:
                if time.time() - self.last_backup > self.config.backup_interval:
                    self._create_backup()
                    self.last_backup = time.time()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
    
    def _create_backup(self):
        """Create backup of knowledge base"""
        backup_path = Path(self.config.persist_directory) / "backups"
        backup_path.mkdir(exist_ok=True)
        
        backup_file = backup_path / f"knowledge_backup_{int(time.time())}.pkl"
        
        backup_data = {
            'knowledge_items': self.knowledge_items,
            'config': self.config,
            'timestamp': time.time(),
            'query_count': self.query_count
        }
        
        with open(backup_file, 'wb') as f:
            pickle.dump(backup_data, f)
        
        logger.info(f"Knowledge base backup created: {backup_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            'total_items': len(self.knowledge_items),
            'vector_db_count': self.vector_db.count(),
            'query_count': self.query_count,
            'avg_unity_relevance': np.mean([item.unity_relevance for item in self.knowledge_items.values()]) if self.knowledge_items else 0.0,
            'avg_consciousness_level': np.mean([item.consciousness_level for item in self.knowledge_items.values()]) if self.knowledge_items else 0.0,
            'knowledge_graph_nodes': len(self.knowledge_graph.nodes) if self.knowledge_graph else 0,
            'last_backup': self.last_backup,
            'uptime_hours': (time.time() - self.last_backup) / 3600.0 if hasattr(self, 'start_time') else 0.0
        }

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_unity_knowledge_base(config: Optional[KnowledgeConfig] = None) -> UnityKnowledgeBase:
    """Factory function to create Unity Knowledge Base"""
    if config is None:
        config = KnowledgeConfig()
    
    kb = UnityKnowledgeBase(config)
    kb.start()
    return kb

# Export public API
__all__ = [
    'KnowledgeConfig',
    'UnityKnowledgeBase',
    'KnowledgeItem',
    'ConceptNode',
    'ConceptRelation',
    'create_unity_knowledge_base'
]