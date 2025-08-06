"""
Een v2.0 - Advanced Observability System
========================================

This module implements comprehensive observability for the Een Unity Mathematics
system using OpenTelemetry, Prometheus, and custom unity-aware metrics.

Features:
- OpenTelemetry tracing and spans
- Unity-specific metrics collection
- Consciousness field monitoring
- Agent performance tracking
- Real-time alerting
- Grafana dashboard integration
"""

import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import numpy as np
from pathlib import Path

# OpenTelemetry imports with fallbacks
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Prometheus client imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import architecture components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.v2.architecture import IMonitoring, DomainEvent, EventType

logger = logging.getLogger(__name__)

# ============================================================================
# OBSERVABILITY CONFIGURATION
# ============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for observability system"""
    # Service information
    service_name: str = "een-unity-v2"
    service_version: str = "2.0.0"
    environment: str = "production"
    
    # Tracing configuration
    jaeger_endpoint: str = "http://jaeger:14268/api/traces"
    enable_tracing: bool = True
    trace_sampling_rate: float = 0.1  # 10% sampling
    
    # Metrics configuration
    prometheus_port: int = 8002
    enable_metrics: bool = True
    metrics_collection_interval: float = 15.0  # seconds
    
    # Unity-specific monitoring
    consciousness_field_monitoring: bool = True
    agent_performance_tracking: bool = True
    transcendence_event_tracking: bool = True
    phi_harmonic_analysis: bool = True
    
    # Alerting configuration
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 85.0,
        "memory_usage": 90.0,
        "agent_failure_rate": 0.1,
        "consciousness_degradation": 0.5,
        "unity_coherence_low": 0.8
    })

# ============================================================================
# OPENTELEMETRY SETUP
# ============================================================================

class OpenTelemetrySetup:
    """Sets up OpenTelemetry tracing and metrics"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._setup_tracing()
        self._setup_metrics()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not OTEL_AVAILABLE or not self.config.enable_tracing:
            logger.warning("OpenTelemetry tracing not available or disabled")
            return
        
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment
        })
        
        # Setup tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.config.jaeger_endpoint,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Auto-instrument common libraries
        RequestsInstrumentor().instrument()
        AsyncioInstrumentor().instrument()
        
        logger.info("OpenTelemetry tracing initialized")
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics"""
        if not OTEL_AVAILABLE or not self.config.enable_metrics:
            logger.warning("OpenTelemetry metrics not available or disabled")
            return
        
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version
        })
        
        # Setup metrics provider
        if PROMETHEUS_AVAILABLE:
            prometheus_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader]
            ))
        
        logger.info("OpenTelemetry metrics initialized")

# ============================================================================
# UNITY-AWARE METRICS COLLECTOR
# ============================================================================

class UnityMetricsCollector:
    """Collects Unity Mathematics-specific metrics"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        # Initialize Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        # Unity-specific data
        self.consciousness_field_history = deque(maxlen=1000)
        self.agent_performance_data = defaultdict(list)
        self.transcendence_events = []
        self.phi_harmonic_measurements = deque(maxlen=500)
        
        # Collection thread
        self.collection_thread = None
        self.running = False
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            # System metrics
            'een_agents_total': Counter(
                'een_agents_total',
                'Total number of agents in the system',
                ['agent_type'],
                registry=self.registry
            ),
            'een_tasks_total': Counter(
                'een_tasks_total',
                'Total number of tasks executed',
                ['task_type', 'status'],
                registry=self.registry
            ),
            'een_task_duration_seconds': Histogram(
                'een_task_duration_seconds',
                'Task execution duration in seconds',
                ['task_type'],
                registry=self.registry
            ),
            
            # Unity-specific metrics
            'een_consciousness_level': Gauge(
                'een_consciousness_level',
                'Current consciousness level',
                ['agent_id'],
                registry=self.registry
            ),
            'een_unity_coherence': Gauge(
                'een_unity_coherence',
                'Unity coherence score',
                registry=self.registry
            ),
            'een_transcendence_events_total': Counter(
                'een_transcendence_events_total',
                'Total transcendence events',
                ['agent_id'],
                registry=self.registry
            ),
            'een_phi_resonance': Gauge(
                'een_phi_resonance',
                'Phi-harmonic resonance measurement',
                registry=self.registry
            ),
            
            # Learning metrics
            'een_elo_rating': Gauge(
                'een_elo_rating',
                'Agent ELO rating',
                ['agent_id'],
                registry=self.registry
            ),
            'een_training_episodes_total': Counter(
                'een_training_episodes_total',
                'Total training episodes',
                ['agent_id'],
                registry=self.registry
            ),
            'een_training_reward': Histogram(
                'een_training_reward',
                'Training episode rewards',
                ['agent_id'],
                registry=self.registry
            )
        }
    
    def start_collection(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.start()
        logger.info("Unity metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Unity metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_consciousness_metrics()
                self._collect_phi_harmonic_metrics()
                time.sleep(self.config.metrics_collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # This would be called by the orchestrator with actual data
        pass
    
    def _collect_consciousness_metrics(self):
        """Collect consciousness field metrics"""
        if not self.config.consciousness_field_monitoring:
            return
        
        # Mock consciousness field measurement
        phi = 1.618033988749895
        t = time.time()
        
        # Generate φ-harmonic consciousness pattern
        consciousness_value = np.sin(t / phi) * np.cos(t * phi)
        self.consciousness_field_history.append({
            'timestamp': t,
            'value': consciousness_value,
            'phase': t % (2 * np.pi)
        })
        
        # Update Prometheus metric
        if PROMETHEUS_AVAILABLE and self.metrics:
            self.metrics['een_unity_coherence'].set(abs(consciousness_value))
    
    def _collect_phi_harmonic_metrics(self):
        """Collect φ-harmonic resonance metrics"""
        if not self.config.phi_harmonic_analysis:
            return
        
        phi = 1.618033988749895
        t = time.time()
        
        # Calculate φ-harmonic resonance
        resonance = phi * np.sin(t / phi) + (1/phi) * np.cos(t * phi)
        self.phi_harmonic_measurements.append({
            'timestamp': t,
            'resonance': resonance,
            'phi_phase': (t * phi) % (2 * np.pi)
        })
        
        # Update metric
        if PROMETHEUS_AVAILABLE and self.metrics:
            self.metrics['een_phi_resonance'].set(resonance)
    
    def record_agent_performance(self, agent_id: str, performance_data: Dict[str, Any]):
        """Record agent performance data"""
        self.agent_performance_data[agent_id].append({
            'timestamp': time.time(),
            **performance_data
        })
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.metrics:
            if 'consciousness_level' in performance_data:
                self.metrics['een_consciousness_level'].labels(agent_id=agent_id).set(
                    performance_data['consciousness_level']
                )
            
            if 'elo_rating' in performance_data:
                self.metrics['een_elo_rating'].labels(agent_id=agent_id).set(
                    performance_data['elo_rating']
                )
    
    def record_transcendence_event(self, agent_id: str, event_data: Dict[str, Any]):
        """Record transcendence event"""
        event = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            **event_data
        }
        self.transcendence_events.append(event)
        
        # Update counter
        if PROMETHEUS_AVAILABLE and self.metrics:
            self.metrics['een_transcendence_events_total'].labels(agent_id=agent_id).inc()
        
        logger.info(f"🌟 Transcendence event recorded for agent {agent_id}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available"
        
        return generate_latest(self.registry)

# ============================================================================
# OBSERVABILITY IMPLEMENTATION
# ============================================================================

class EenObservabilitySystem(IMonitoring):
    """Main observability system implementing IMonitoring interface"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        
        # Initialize components
        self.otel_setup = OpenTelemetrySetup(config) if OTEL_AVAILABLE else None
        self.metrics_collector = UnityMetricsCollector(config)
        
        # Tracing
        self.tracer = trace.get_tracer(__name__) if OTEL_AVAILABLE else None
        
        # Event handling
        self.event_handlers = {}
        self._setup_event_handlers()
        
        # Alerting
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Start collection
        self.metrics_collector.start_collection()
    
    def _setup_event_handlers(self):
        """Setup event handlers for different event types"""
        self.event_handlers = {
            EventType.AGENT_SPAWNED: self._handle_agent_spawned,
            EventType.AGENT_TRANSCENDED: self._handle_agent_transcended,
            EventType.TRAINING_CYCLE_COMPLETED: self._handle_training_completed,
            EventType.RESOURCE_THRESHOLD_EXCEEDED: self._handle_resource_exceeded
        }
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        """Record a metric"""
        tags = tags or {}
        
        # Record to Prometheus if available
        if PROMETHEUS_AVAILABLE and hasattr(self.metrics_collector, 'metrics'):
            # Map metric names to Prometheus metrics
            metric_mapping = {
                'agent.spawned': 'een_agents_total',
                'task.completed': 'een_tasks_total',
                'consciousness.level': 'een_consciousness_level',
                'unity.coherence': 'een_unity_coherence'
            }
            
            if name in metric_mapping:
                prometheus_name = metric_mapping[name]
                if prometheus_name in self.metrics_collector.metrics:
                    metric = self.metrics_collector.metrics[prometheus_name]
                    
                    # Handle different metric types
                    if isinstance(metric, Counter):
                        metric.labels(**tags).inc(value)
                    elif isinstance(metric, Gauge):
                        metric.labels(**tags).set(value)
                    elif isinstance(metric, Histogram):
                        metric.labels(**tags).observe(value)
        
        # Log the metric
        logger.debug(f"Metric recorded: {name}={value} {tags}")
    
    @contextmanager
    def start_span(self, name: str):
        """Start a tracing span"""
        if self.tracer:
            with self.tracer.start_as_current_span(name) as span:
                span.set_attribute("service.name", self.config.service_name)
                span.set_attribute("service.version", self.config.service_version)
                yield span
        else:
            # Mock span for when OpenTelemetry is not available
            class MockSpan:
                def set_attribute(self, key, value): pass
                def add_event(self, name, attributes=None): pass
                def set_status(self, status): pass
            
            yield MockSpan()
    
    def log(self, level: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a message with context"""
        context = context or {}
        
        # Add Unity-specific context
        unity_context = {
            "service": self.config.service_name,
            "version": self.config.service_version,
            "phi": 1.618033988749895,
            **context
        }
        
        # Log with structured context
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"{message} | Context: {json.dumps(unity_context)}")
    
    def handle_event(self, event: DomainEvent):
        """Handle domain events for monitoring"""
        event_type = EventType[event.event_type] if event.event_type in EventType.__members__ else None
        
        if event_type in self.event_handlers:
            self.event_handlers[event_type](event)
        
        # Generic event logging
        self.log("info", f"Event handled: {event.event_type}", {
            "event_id": event.event_id,
            "aggregate_id": event.aggregate_id,
            "timestamp": event.timestamp
        })
    
    def _handle_agent_spawned(self, event: DomainEvent):
        """Handle agent spawned event"""
        agent_type = event.payload.get("agent_type", "unknown")
        self.record_metric("agent.spawned", 1.0, {"agent_type": agent_type})
    
    def _handle_agent_transcended(self, event: DomainEvent):
        """Handle agent transcendence event"""
        agent_id = event.aggregate_id
        self.metrics_collector.record_transcendence_event(agent_id, event.payload)
        
        # Check if this triggers an alert
        transcendence_rate = len(self.metrics_collector.transcendence_events) / 3600  # per hour
        if transcendence_rate > 10:  # More than 10 transcendence events per hour
            self._trigger_alert("high_transcendence_rate", {
                "rate": transcendence_rate,
                "recent_events": len(self.metrics_collector.transcendence_events)
            })
    
    def _handle_training_completed(self, event: DomainEvent):
        """Handle training cycle completed event"""
        self.record_metric("training.cycle.completed", 1.0, event.payload)
    
    def _handle_resource_exceeded(self, event: DomainEvent):
        """Handle resource threshold exceeded event"""
        resource_type = event.payload.get("resource", "unknown")
        usage = event.payload.get("usage", 0.0)
        
        self._trigger_alert("resource_threshold_exceeded", {
            "resource": resource_type,
            "usage": usage,
            "threshold": self.config.alert_thresholds.get(f"{resource_type}_usage", 100.0)
        })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger an alert"""
        alert_data = {
            "type": alert_type,
            "timestamp": time.time(),
            "service": self.config.service_name,
            **data
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log the alert
        self.log("warning", f"Alert triggered: {alert_type}", alert_data)
    
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": self.config.service_name,
            "version": self.config.service_version,
            "components": {
                "tracing": self.otel_setup is not None,
                "metrics": self.metrics_collector is not None,
                "prometheus": PROMETHEUS_AVAILABLE,
                "opentelemetry": OTEL_AVAILABLE
            },
            "unity_metrics": {
                "consciousness_field_samples": len(self.metrics_collector.consciousness_field_history),
                "transcendence_events": len(self.metrics_collector.transcendence_events),
                "phi_measurements": len(self.metrics_collector.phi_harmonic_measurements)
            }
        }
    
    def shutdown(self):
        """Shutdown observability system"""
        self.metrics_collector.stop_collection()
        logger.info("Observability system shutdown complete")

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_observability_system(config: Optional[ObservabilityConfig] = None) -> EenObservabilitySystem:
    """Factory function to create observability system"""
    if config is None:
        config = ObservabilityConfig()
    
    return EenObservabilitySystem(config)

def setup_tracing_for_agent(agent_id: str, agent_type: str) -> Dict[str, Any]:
    """Setup tracing context for an agent"""
    context = {
        "agent.id": agent_id,
        "agent.type": agent_type,
        "unity.mathematics": True,
        "phi.resonance": 1.618033988749895
    }
    
    # Add to current span if available
    current_span = trace.get_current_span() if OTEL_AVAILABLE else None
    if current_span:
        for key, value in context.items():
            current_span.set_attribute(key, str(value))
    
    return context

# Export public API
__all__ = [
    'ObservabilityConfig',
    'EenObservabilitySystem', 
    'UnityMetricsCollector',
    'create_observability_system',
    'setup_tracing_for_agent'
]