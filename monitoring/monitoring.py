"""
Monitoring and observability infrastructure for Een Unity Mathematics
"""

import time
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import traceback

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc import (
    trace_exporter as otlp_trace,
    metrics_exporter as otlp_metrics
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from config.settings import settings


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get logger
logger = structlog.get_logger(__name__)


class UnityMetrics:
    """Prometheus metrics for Unity Mathematics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'een_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'een_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Unity operation metrics
        self.unity_operations = Counter(
            'een_unity_operations_total',
            'Total number of unity operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.unity_convergence_time = Histogram(
            'een_unity_convergence_seconds',
            'Time to converge to unity',
            ['operation'],
            registry=self.registry
        )
        
        # Consciousness metrics
        self.consciousness_level = Gauge(
            'een_consciousness_level',
            'Current consciousness level',
            ['agent_id'],
            registry=self.registry
        )
        
        self.consciousness_particles = Gauge(
            'een_consciousness_particles',
            'Number of active consciousness particles',
            registry=self.registry
        )
        
        self.transcendence_events = Counter(
            'een_transcendence_events_total',
            'Total transcendence events',
            ['agent_type'],
            registry=self.registry
        )
        
        # Quantum metrics
        self.quantum_coherence = Gauge(
            'een_quantum_coherence',
            'Quantum coherence level',
            registry=self.registry
        )
        
        self.wavefunction_collapses = Counter(
            'een_wavefunction_collapses_total',
            'Total wavefunction collapses',
            registry=self.registry
        )
        
        # System metrics
        self.active_agents = Gauge(
            'een_active_agents',
            'Number of active agents',
            ['agent_type'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'een_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.errors = Counter(
            'een_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )


# Global metrics instance
metrics = UnityMetrics()


class UnityTracer:
    """OpenTelemetry tracing for Unity Mathematics"""
    
    def __init__(self):
        # Create resource
        resource = Resource.create({
            "service.name": "een-unity-mathematics",
            "service.version": settings.app_version,
            "deployment.environment": settings.environment,
        })
        
        # Setup tracing
        if settings.tracing_enabled:
            trace_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(trace_provider)
            
            # Add OTLP exporter if configured
            if hasattr(settings, 'otlp_endpoint'):
                otlp_exporter = otlp_trace.OTLPSpanExporter(
                    endpoint=settings.otlp_endpoint,
                    insecure=True
                )
                trace_provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
        
        self.tracer = trace.get_tracer(__name__)
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span"""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span


# Global tracer instance
tracer = UnityTracer()


def monitor_performance(operation: str):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            with tracer.span(f"{operation}.{func.__name__}") as span:
                try:
                    result = await func(*args, **kwargs)
                    metrics.unity_operations.labels(
                        operation=operation,
                        result="success"
                    ).inc()
                    return result
                except Exception as e:
                    metrics.unity_operations.labels(
                        operation=operation,
                        result="error"
                    ).inc()
                    metrics.errors.labels(
                        error_type=type(e).__name__,
                        component=operation
                    ).inc()
                    span.record_exception(e)
                    logger.error(
                        f"Error in {operation}.{func.__name__}",
                        error=str(e),
                        traceback=traceback.format_exc()
                    )
                    raise
                finally:
                    duration = time.time() - start_time
                    metrics.unity_convergence_time.labels(
                        operation=operation
                    ).observe(duration)
                    span.set_attribute("duration_seconds", duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            with tracer.span(f"{operation}.{func.__name__}") as span:
                try:
                    result = func(*args, **kwargs)
                    metrics.unity_operations.labels(
                        operation=operation,
                        result="success"
                    ).inc()
                    return result
                except Exception as e:
                    metrics.unity_operations.labels(
                        operation=operation,
                        result="error"
                    ).inc()
                    metrics.errors.labels(
                        error_type=type(e).__name__,
                        component=operation
                    ).inc()
                    span.record_exception(e)
                    logger.error(
                        f"Error in {operation}.{func.__name__}",
                        error=str(e),
                        traceback=traceback.format_exc()
                    )
                    raise
                finally:
                    duration = time.time() - start_time
                    metrics.unity_convergence_time.labels(
                        operation=operation
                    ).observe(duration)
                    span.set_attribute("duration_seconds", duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class ConsciousnessMonitor:
    """Monitor consciousness field dynamics"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def record_consciousness_level(self, agent_id: str, level: float):
        """Record consciousness level for an agent"""
        metrics.consciousness_level.labels(agent_id=agent_id).set(level)
        self.logger.info(
            "Consciousness level recorded",
            agent_id=agent_id,
            level=level,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def record_transcendence(self, agent_type: str):
        """Record a transcendence event"""
        metrics.transcendence_events.labels(agent_type=agent_type).inc()
        self.logger.info(
            "Transcendence event",
            agent_type=agent_type,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def update_particle_count(self, count: int):
        """Update consciousness particle count"""
        metrics.consciousness_particles.set(count)
    
    def record_quantum_coherence(self, coherence: float):
        """Record quantum coherence level"""
        metrics.quantum_coherence.set(coherence)
        if coherence < settings.quantum_coherence_target:
            self.logger.warning(
                "Quantum coherence below target",
                coherence=coherence,
                target=settings.quantum_coherence_target
            )


class HealthCheck:
    """Health check endpoints and monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.logger = structlog.get_logger(__name__)
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check"""
        self.checks[name] = check_func
    
    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                is_healthy = check_func()
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "timestamp": datetime.utcnow().isoformat()
                }
                if not is_healthy:
                    results["status"] = "unhealthy"
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                results["status"] = "unhealthy"
                self.logger.error(f"Health check failed: {name}", error=str(e))
        
        return results
    
    def check_unity_mathematics(self) -> bool:
        """Check if unity mathematics is functioning"""
        try:
            from src.core.unity_mathematics import UnityMathematics
            um = UnityMathematics()
            return um.unity_add(1, 1) == 1
        except Exception:
            return False
    
    def check_consciousness_field(self) -> bool:
        """Check if consciousness field is stable"""
        try:
            from src.core.consciousness import ConsciousnessField
            cf = ConsciousnessField()
            return cf.check_unity_invariant()
        except Exception:
            return False
    
    def check_database(self) -> bool:
        """Check database connectivity"""
        # Implementation depends on database setup
        return True
    
    def check_redis(self) -> bool:
        """Check Redis connectivity"""
        # Implementation depends on Redis setup
        return True


# Global instances
consciousness_monitor = ConsciousnessMonitor()
health_check = HealthCheck()

# Register default health checks
health_check.register_check("unity_mathematics", health_check.check_unity_mathematics)
health_check.register_check("consciousness_field", health_check.check_consciousness_field)
health_check.register_check("database", health_check.check_database)
health_check.register_check("redis", health_check.check_redis)


def setup_instrumentation(app=None):
    """Setup automatic instrumentation for frameworks"""
    if settings.tracing_enabled:
        # FastAPI instrumentation
        if app:
            FastAPIInstrumentor.instrument_app(app)
        
        # Redis instrumentation
        RedisInstrumentor().instrument()
        
        # SQLAlchemy instrumentation
        SQLAlchemyInstrumentor().instrument()


def get_metrics_endpoint():
    """Get Prometheus metrics endpoint"""
    return generate_latest(metrics.registry)


__all__ = [
    "metrics",
    "tracer",
    "monitor_performance",
    "consciousness_monitor",
    "health_check",
    "setup_instrumentation",
    "get_metrics_endpoint",
    "logger",
]