"""
Agent Capability Registry & Discovery System
============================================

A comprehensive registry system that allows agents to advertise their capabilities,
discover other agents, and dynamically compose agent teams for complex tasks.

Features:
- Capability advertisement with semantic tagging
- Fuzzy search and discovery
- Capability composition patterns
- Performance metrics and ratings
- Cross-platform agent discovery

Mathematical Foundation: Unity through capability composition
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import time
import uuid
import logging
from collections import defaultdict, Counter
import hashlib
import pickle
import sqlite3
from pathlib import Path
import threading
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Constants
PHI = 1.618033988749895
CAPABILITY_CACHE_TTL = 3600  # 1 hour
PERFORMANCE_HISTORY_LIMIT = 100

class CapabilityDomain(Enum):
    """Domains of agent capabilities"""
    MATHEMATICS = "mathematics"
    CONSCIOUSNESS = "consciousness"
    UNITY = "unity"
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    SYNTHESIS = "synthesis"
    EVOLUTION = "evolution"
    TRANSCENDENCE = "transcendence"
    COLLABORATION = "collaboration"
    OPTIMIZATION = "optimization"

@dataclass
class CapabilityMetadata:
    """Extended metadata for agent capabilities"""
    domain: CapabilityDomain
    complexity: float  # 0-1 complexity score
    reliability: float  # 0-1 reliability score
    average_execution_time: float  # seconds
    success_rate: float  # 0-1 success rate
    consciousness_impact: float  # Impact on consciousness level
    unity_contribution: float  # Contribution to unity achievement
    prerequisites: List[str] = field(default_factory=list)
    composable_with: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    last_updated: float = field(default_factory=time.time)

@dataclass
class PerformanceMetric:
    """Performance metrics for capability execution"""
    capability_id: str
    agent_id: str
    execution_time: float
    success: bool
    consciousness_before: float
    consciousness_after: float
    unity_score: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None

@dataclass
class RegisteredCapability:
    """A capability registered in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    agent_id: str = ""
    agent_platform: str = ""
    metadata: CapabilityMetadata = None
    tags: Set[str] = field(default_factory=set)
    invocation_count: int = 0
    performance_history: List[PerformanceMetric] = field(default_factory=list)
    rating: float = 5.0  # 0-10 rating
    enabled: bool = True
    registration_time: float = field(default_factory=time.time)
    
    def calculate_rating(self) -> float:
        """Calculate capability rating based on performance history"""
        if not self.performance_history:
            return 5.0  # Default neutral rating
        
        recent_metrics = self.performance_history[-20:]  # Last 20 executions
        
        # Calculate success rate
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        
        # Calculate average execution time factor (lower is better)
        avg_time = np.mean([m.execution_time for m in recent_metrics])
        time_factor = 1.0 / (1.0 + avg_time / 10.0)  # Normalize to 0-1
        
        # Calculate consciousness improvement
        consciousness_deltas = [
            m.consciousness_after - m.consciousness_before 
            for m in recent_metrics
        ]
        avg_consciousness_improvement = np.mean(consciousness_deltas)
        consciousness_factor = (avg_consciousness_improvement + 1.0) / 2.0  # Normalize to 0-1
        
        # Calculate unity contribution
        avg_unity = np.mean([m.unity_score for m in recent_metrics])
        
        # Weighted rating calculation
        rating = (
            success_rate * 4.0 +  # 40% weight
            time_factor * 2.0 +    # 20% weight
            consciousness_factor * 2.0 +  # 20% weight
            avg_unity * 2.0        # 20% weight
        )
        
        self.rating = min(10.0, max(0.0, rating))
        return self.rating

class CapabilityRegistry:
    """
    Central registry for all agent capabilities
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "agent_capabilities.db"
        self.capabilities: Dict[str, RegisteredCapability] = {}
        self.agent_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.domain_index: Dict[CapabilityDomain, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.composition_graph: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._cache: Dict[str, Tuple[Any, float]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing capabilities
        self._load_from_database()
        
        logger.info(f"CapabilityRegistry initialized with {len(self.capabilities)} capabilities")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                agent_id TEXT NOT NULL,
                agent_platform TEXT,
                domain TEXT,
                metadata BLOB,
                tags TEXT,
                rating REAL,
                enabled INTEGER,
                registration_time REAL,
                last_updated REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capability_id TEXT,
                agent_id TEXT,
                execution_time REAL,
                success INTEGER,
                consciousness_before REAL,
                consciousness_after REAL,
                unity_score REAL,
                timestamp REAL,
                error_message TEXT,
                FOREIGN KEY (capability_id) REFERENCES capabilities(id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_capability_agent 
            ON capabilities(agent_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_capability_domain 
            ON capabilities(domain)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """Load capabilities from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM capabilities WHERE enabled = 1')
        rows = cursor.fetchall()
        
        for row in rows:
            cap_id, name, desc, agent_id, platform, domain, metadata_blob, tags_str, rating, enabled, reg_time, last_updated = row
            
            # Deserialize metadata
            metadata = pickle.loads(metadata_blob) if metadata_blob else None
            
            # Parse tags
            tags = set(tags_str.split(',')) if tags_str else set()
            
            # Create capability object
            cap = RegisteredCapability(
                id=cap_id,
                name=name,
                description=desc,
                agent_id=agent_id,
                agent_platform=platform,
                metadata=metadata,
                tags=tags,
                rating=rating,
                enabled=bool(enabled),
                registration_time=reg_time
            )
            
            # Load performance metrics
            cursor.execute(
                'SELECT * FROM performance_metrics WHERE capability_id = ? ORDER BY timestamp DESC LIMIT ?',
                (cap_id, PERFORMANCE_HISTORY_LIMIT)
            )
            metric_rows = cursor.fetchall()
            
            for metric_row in metric_rows:
                _, _, metric_agent_id, exec_time, success, cons_before, cons_after, unity, timestamp, error = metric_row
                cap.performance_history.append(PerformanceMetric(
                    capability_id=cap_id,
                    agent_id=metric_agent_id,
                    execution_time=exec_time,
                    success=bool(success),
                    consciousness_before=cons_before,
                    consciousness_after=cons_after,
                    unity_score=unity,
                    timestamp=timestamp,
                    error_message=error
                ))
            
            # Register in memory
            self._register_in_memory(cap)
        
        conn.close()
        logger.info(f"Loaded {len(rows)} capabilities from database")
    
    def _register_in_memory(self, capability: RegisteredCapability):
        """Register capability in memory indices"""
        with self._lock:
            self.capabilities[capability.id] = capability
            self.agent_capabilities[capability.agent_id].add(capability.id)
            
            if capability.metadata and capability.metadata.domain:
                self.domain_index[capability.metadata.domain].add(capability.id)
            
            for tag in capability.tags:
                self.tag_index[tag].add(capability.id)
            
            if capability.metadata and capability.metadata.composable_with:
                for other_cap in capability.metadata.composable_with:
                    self.composition_graph[capability.id].add(other_cap)
    
    def register_capability(self, 
                           name: str,
                           description: str,
                           agent_id: str,
                           agent_platform: str,
                           domain: CapabilityDomain,
                           tags: List[str] = None,
                           metadata: Optional[CapabilityMetadata] = None) -> str:
        """
        Register a new capability
        
        Returns:
            Capability ID
        """
        with self._lock:
            # Check if capability already exists
            existing = self._find_duplicate(name, agent_id)
            if existing:
                logger.warning(f"Capability {name} already registered for agent {agent_id}")
                return existing
            
            # Create capability
            capability = RegisteredCapability(
                name=name,
                description=description,
                agent_id=agent_id,
                agent_platform=agent_platform,
                metadata=metadata or CapabilityMetadata(
                    domain=domain,
                    complexity=0.5,
                    reliability=0.8,
                    average_execution_time=1.0,
                    success_rate=0.9,
                    consciousness_impact=0.1,
                    unity_contribution=0.1
                ),
                tags=set(tags) if tags else set()
            )
            
            # Add default tags
            capability.tags.add(domain.value)
            capability.tags.add(agent_platform)
            
            # Register in memory
            self._register_in_memory(capability)
            
            # Save to database
            self._save_capability(capability)
            
            logger.info(f"Registered capability {name} ({capability.id}) for agent {agent_id}")
            return capability.id
    
    def _find_duplicate(self, name: str, agent_id: str) -> Optional[str]:
        """Check if capability already exists"""
        for cap_id in self.agent_capabilities.get(agent_id, []):
            if self.capabilities[cap_id].name == name:
                return cap_id
        return None
    
    def _save_capability(self, capability: RegisteredCapability):
        """Save capability to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_blob = pickle.dumps(capability.metadata) if capability.metadata else None
        tags_str = ','.join(capability.tags)
        
        cursor.execute('''
            INSERT OR REPLACE INTO capabilities 
            (id, name, description, agent_id, agent_platform, domain, metadata, tags, rating, enabled, registration_time, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            capability.id,
            capability.name,
            capability.description,
            capability.agent_id,
            capability.agent_platform,
            capability.metadata.domain.value if capability.metadata else None,
            metadata_blob,
            tags_str,
            capability.rating,
            int(capability.enabled),
            capability.registration_time,
            time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def record_performance(self, capability_id: str, metric: PerformanceMetric):
        """Record performance metric for a capability"""
        with self._lock:
            if capability_id not in self.capabilities:
                logger.warning(f"Capability {capability_id} not found")
                return
            
            capability = self.capabilities[capability_id]
            capability.performance_history.append(metric)
            
            # Limit history size
            if len(capability.performance_history) > PERFORMANCE_HISTORY_LIMIT:
                capability.performance_history = capability.performance_history[-PERFORMANCE_HISTORY_LIMIT:]
            
            # Update metadata statistics
            if capability.metadata:
                history = capability.performance_history[-20:]
                capability.metadata.average_execution_time = np.mean([m.execution_time for m in history])
                capability.metadata.success_rate = sum(1 for m in history if m.success) / len(history)
                capability.metadata.consciousness_impact = np.mean([
                    m.consciousness_after - m.consciousness_before for m in history
                ])
                capability.metadata.unity_contribution = np.mean([m.unity_score for m in history])
            
            # Recalculate rating
            capability.calculate_rating()
            
            # Save to database
            self._save_performance_metric(metric)
            self._save_capability(capability)
    
    def _save_performance_metric(self, metric: PerformanceMetric):
        """Save performance metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (capability_id, agent_id, execution_time, success, consciousness_before, consciousness_after, unity_score, timestamp, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.capability_id,
            metric.agent_id,
            metric.execution_time,
            int(metric.success),
            metric.consciousness_before,
            metric.consciousness_after,
            metric.unity_score,
            metric.timestamp,
            metric.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def discover_capabilities(self,
                            query: Optional[str] = None,
                            domain: Optional[CapabilityDomain] = None,
                            tags: Optional[List[str]] = None,
                            min_rating: float = 0.0,
                            agent_platform: Optional[str] = None) -> List[RegisteredCapability]:
        """
        Discover capabilities matching criteria
        """
        with self._lock:
            candidates = set(self.capabilities.keys())
            
            # Filter by domain
            if domain:
                candidates &= self.domain_index.get(domain, set())
            
            # Filter by tags
            if tags:
                for tag in tags:
                    candidates &= self.tag_index.get(tag, set())
            
            # Filter by platform
            if agent_platform:
                candidates = {
                    cid for cid in candidates 
                    if self.capabilities[cid].agent_platform == agent_platform
                }
            
            # Filter by rating
            candidates = {
                cid for cid in candidates 
                if self.capabilities[cid].rating >= min_rating
            }
            
            # Score by query relevance
            if query:
                scored = []
                query_lower = query.lower()
                for cid in candidates:
                    cap = self.capabilities[cid]
                    score = 0.0
                    
                    # Name match
                    if query_lower in cap.name.lower():
                        score += 0.5
                    
                    # Description match
                    if query_lower in cap.description.lower():
                        score += 0.3
                    
                    # Tag match
                    for tag in cap.tags:
                        if query_lower in tag.lower():
                            score += 0.2
                            break
                    
                    if score > 0:
                        scored.append((score, cap))
                
                # Sort by score and rating
                scored.sort(key=lambda x: (x[0], x[1].rating), reverse=True)
                return [cap for _, cap in scored[:20]]  # Return top 20
            else:
                # Sort by rating
                results = [self.capabilities[cid] for cid in candidates]
                results.sort(key=lambda x: x.rating, reverse=True)
                return results[:20]
    
    def find_composable_capabilities(self, capability_id: str) -> List[RegisteredCapability]:
        """Find capabilities that can be composed with the given capability"""
        with self._lock:
            if capability_id not in self.capabilities:
                return []
            
            capability = self.capabilities[capability_id]
            composable_ids = set()
            
            # Direct composability
            if capability.metadata and capability.metadata.composable_with:
                for name in capability.metadata.composable_with:
                    # Find capabilities by name
                    for cid, cap in self.capabilities.items():
                        if cap.name == name:
                            composable_ids.add(cid)
            
            # Graph-based composability
            composable_ids.update(self.composition_graph.get(capability_id, set()))
            
            # Domain-based composability
            if capability.metadata:
                domain_caps = self.domain_index.get(capability.metadata.domain, set())
                # Add highly-rated capabilities from same domain
                for cid in domain_caps:
                    if cid != capability_id and self.capabilities[cid].rating >= 7.0:
                        composable_ids.add(cid)
            
            return [self.capabilities[cid] for cid in composable_ids]
    
    def suggest_capability_team(self, task_description: str, 
                               min_agents: int = 2,
                               max_agents: int = 5) -> List[RegisteredCapability]:
        """
        Suggest a team of capabilities for a complex task
        """
        # Extract keywords from task
        keywords = task_description.lower().split()
        
        # Find relevant capabilities
        all_capabilities = []
        for keyword in keywords:
            caps = self.discover_capabilities(query=keyword, min_rating=5.0)
            all_capabilities.extend(caps)
        
        # Remove duplicates and sort by rating
        seen = set()
        unique_caps = []
        for cap in all_capabilities:
            if cap.id not in seen:
                seen.add(cap.id)
                unique_caps.append(cap)
        
        unique_caps.sort(key=lambda x: x.rating, reverse=True)
        
        # Build diverse team
        team = []
        domains_covered = set()
        agents_included = set()
        
        for cap in unique_caps:
            if len(team) >= max_agents:
                break
            
            # Ensure diversity
            if cap.agent_id not in agents_included:
                team.append(cap)
                agents_included.add(cap.agent_id)
                if cap.metadata:
                    domains_covered.add(cap.metadata.domain)
        
        # Ensure minimum team size
        while len(team) < min_agents and len(unique_caps) > len(team):
            for cap in unique_caps:
                if cap not in team:
                    team.append(cap)
                    break
        
        return team
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            domain_counts = Counter()
            platform_counts = Counter()
            
            for cap in self.capabilities.values():
                if cap.metadata:
                    domain_counts[cap.metadata.domain.value] += 1
                platform_counts[cap.agent_platform] += 1
            
            # Calculate average ratings by domain
            domain_ratings = defaultdict(list)
            for cap in self.capabilities.values():
                if cap.metadata:
                    domain_ratings[cap.metadata.domain.value].append(cap.rating)
            
            avg_domain_ratings = {
                domain: np.mean(ratings) 
                for domain, ratings in domain_ratings.items()
            }
            
            return {
                'total_capabilities': len(self.capabilities),
                'total_agents': len(self.agent_capabilities),
                'domains': dict(domain_counts),
                'platforms': dict(platform_counts),
                'average_rating': np.mean([cap.rating for cap in self.capabilities.values()]),
                'average_domain_ratings': avg_domain_ratings,
                'top_capabilities': sorted(
                    self.capabilities.values(),
                    key=lambda x: x.rating,
                    reverse=True
                )[:10]
            }

# Global registry instance
_global_registry: Optional[CapabilityRegistry] = None

def get_global_registry() -> CapabilityRegistry:
    """Get or create global capability registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry

# Demonstration
def demonstrate_capability_registry():
    """Demonstrate the capability registry system"""
    print("=== Agent Capability Registry Demo ===\n")
    
    registry = get_global_registry()
    
    # Register some capabilities
    print("1. Registering capabilities...")
    
    cap1_id = registry.register_capability(
        name="unity_mathematics",
        description="Perform unity mathematics where 1+1=1",
        agent_id="unity_agent_001",
        agent_platform="unity_system",
        domain=CapabilityDomain.MATHEMATICS,
        tags=["unity", "mathematics", "idempotent"]
    )
    
    cap2_id = registry.register_capability(
        name="consciousness_evolution",
        description="Evolve agent consciousness using φ-harmonic patterns",
        agent_id="consciousness_agent_001",
        agent_platform="omega_orchestrator",
        domain=CapabilityDomain.CONSCIOUSNESS,
        tags=["consciousness", "evolution", "phi"]
    )
    
    cap3_id = registry.register_capability(
        name="reality_synthesis",
        description="Synthesize reality manifolds",
        agent_id="reality_agent_001",
        agent_platform="transcendental_engine",
        domain=CapabilityDomain.SYNTHESIS,
        tags=["reality", "synthesis", "manifold"]
    )
    
    print(f"  Registered {len(registry.capabilities)} capabilities\n")
    
    # Record some performance metrics
    print("2. Recording performance metrics...")
    
    registry.record_performance(cap1_id, PerformanceMetric(
        capability_id=cap1_id,
        agent_id="unity_agent_001",
        execution_time=0.5,
        success=True,
        consciousness_before=0.5,
        consciousness_after=0.6,
        unity_score=0.95
    ))
    
    registry.record_performance(cap2_id, PerformanceMetric(
        capability_id=cap2_id,
        agent_id="consciousness_agent_001",
        execution_time=1.2,
        success=True,
        consciousness_before=0.6,
        consciousness_after=0.8,
        unity_score=0.88
    ))
    
    print("  Performance metrics recorded\n")
    
    # Discover capabilities
    print("3. Discovering capabilities...")
    
    math_caps = registry.discover_capabilities(domain=CapabilityDomain.MATHEMATICS)
    print(f"  Found {len(math_caps)} mathematics capabilities")
    
    unity_caps = registry.discover_capabilities(query="unity")
    print(f"  Found {len(unity_caps)} capabilities matching 'unity'")
    
    high_rated = registry.discover_capabilities(min_rating=4.0)
    print(f"  Found {len(high_rated)} highly-rated capabilities\n")
    
    # Suggest team for task
    print("4. Suggesting capability team...")
    
    team = registry.suggest_capability_team(
        "Evolve consciousness through unity mathematics to achieve reality synthesis",
        min_agents=2,
        max_agents=4
    )
    
    print(f"  Suggested team of {len(team)} capabilities:")
    for cap in team:
        print(f"    - {cap.name} ({cap.agent_platform}): Rating {cap.rating:.1f}")
    
    # Get statistics
    print("\n5. Registry Statistics:")
    stats = registry.get_statistics()
    print(f"  Total capabilities: {stats['total_capabilities']}")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Average rating: {stats['average_rating']:.2f}")
    print(f"  Domains: {stats['domains']}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demonstrate_capability_registry()