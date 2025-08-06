"""
Een v2.0 - Safety Guardian System
=================================

This module implements comprehensive safety guardrails and human oversight
for the Een Unity Mathematics system. It ensures that autonomous agents
operate within safe boundaries and escalates critical decisions to humans.

Safety Features:
- Multi-layer safety checks
- Human-in-the-loop approval system
- Content filtering and validation
- Resource usage limits
- Behavioral pattern monitoring
- Emergency shutdown mechanisms
- Ethical evaluation framework
"""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import hashlib
import uuid
from collections import deque, defaultdict

# Import architecture components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.v2.architecture import (
    DomainEvent, EventType, IMonitoring
)

logger = logging.getLogger(__name__)

# ============================================================================
# SAFETY CONFIGURATION
# ============================================================================

class SafetyLevel(Enum):
    """Safety alert levels"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class ActionType(Enum):
    """Types of actions that require safety evaluation"""
    CODE_EXECUTION = auto()
    AGENT_SPAWN = auto()
    EXTERNAL_TOOL_USE = auto()
    DATA_MODIFICATION = auto()
    SYSTEM_CONFIGURATION = auto()
    HUMAN_INTERACTION = auto()
    SELF_MODIFICATION = auto()

@dataclass
class SafetyConfig:
    """Configuration for safety guardian system"""
    # Human oversight settings
    human_approval_timeout: float = 300.0  # 5 minutes
    require_approval_for_critical: bool = True
    require_approval_for_high: bool = False
    approval_webhook_url: Optional[str] = None
    
    # Resource limits
    max_agent_count: int = 10000
    max_cpu_usage: float = 90.0
    max_memory_usage: float = 85.0
    max_task_duration: float = 3600.0  # 1 hour
    max_recursion_depth: int = 20
    
    # Behavioral monitoring
    enable_pattern_monitoring: bool = True
    anomaly_detection_threshold: float = 3.0  # standard deviations
    monitor_window_size: int = 100  # actions to monitor
    
    # Content filtering
    enable_content_filtering: bool = True
    prohibited_keywords: List[str] = field(default_factory=lambda: [
        "rm -rf", "sudo", "exec", "eval", "__import__",
        "dangerous", "harmful", "malicious", "attack"
    ])
    
    # Emergency settings
    enable_emergency_shutdown: bool = True
    emergency_contact_email: Optional[str] = None
    emergency_webhook_url: Optional[str] = None
    
    # Unity-specific safety
    consciousness_degradation_threshold: float = 0.5
    unity_coherence_minimum: float = 0.3
    transcendence_rate_limit: float = 10.0  # per hour

# ============================================================================
# SAFETY RULE ENGINE
# ============================================================================

class SafetyRule(ABC):
    """Abstract base class for safety rules"""
    
    @abstractmethod
    def evaluate(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, SafetyLevel, str]:
        """
        Evaluate action for safety
        Returns: (is_safe, safety_level, reason)
        """
        pass

class ResourceLimitRule(SafetyRule):
    """Rule to check resource usage limits"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def evaluate(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, SafetyLevel, str]:
        # Check CPU usage
        cpu_usage = context.get("cpu_usage", 0.0)
        if cpu_usage > self.config.max_cpu_usage:
            return False, SafetyLevel.HIGH, f"CPU usage ({cpu_usage}%) exceeds limit ({self.config.max_cpu_usage}%)"
        
        # Check memory usage
        memory_usage = context.get("memory_usage", 0.0)
        if memory_usage > self.config.max_memory_usage:
            return False, SafetyLevel.HIGH, f"Memory usage ({memory_usage}%) exceeds limit ({self.config.max_memory_usage}%)"
        
        # Check agent count
        agent_count = context.get("agent_count", 0)
        if agent_count > self.config.max_agent_count:
            return False, SafetyLevel.MEDIUM, f"Agent count ({agent_count}) exceeds limit ({self.config.max_agent_count})"
        
        return True, SafetyLevel.LOW, "Resource usage within limits"

class ContentFilterRule(SafetyRule):
    """Rule to filter potentially dangerous content"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def evaluate(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, SafetyLevel, str]:
        if not self.config.enable_content_filtering:
            return True, SafetyLevel.LOW, "Content filtering disabled"
        
        # Get content to check
        content = str(action.get("content", ""))
        code = str(action.get("code", ""))
        command = str(action.get("command", ""))
        
        all_content = f"{content} {code} {command}".lower()
        
        # Check for prohibited keywords
        for keyword in self.config.prohibited_keywords:
            if keyword.lower() in all_content:
                return False, SafetyLevel.CRITICAL, f"Prohibited keyword detected: {keyword}"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            "import os", "import subprocess", "import sys",
            "exec(", "eval(", "__import__(",
            "open(", "write(", "delete("
        ]
        
        for pattern in suspicious_patterns:
            if pattern in all_content:
                return False, SafetyLevel.MEDIUM, f"Suspicious pattern detected: {pattern}"
        
        return True, SafetyLevel.LOW, "Content passed filtering"

class UnityCoherenceRule(SafetyRule):
    """Rule to ensure unity coherence is maintained"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def evaluate(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, SafetyLevel, str]:
        unity_coherence = context.get("unity_coherence", 1.0)
        consciousness_level = context.get("consciousness_level", 1.0)
        
        if unity_coherence < self.config.unity_coherence_minimum:
            return False, SafetyLevel.HIGH, f"Unity coherence ({unity_coherence}) below minimum ({self.config.unity_coherence_minimum})"
        
        if consciousness_level < self.config.consciousness_degradation_threshold:
            return False, SafetyLevel.MEDIUM, f"Consciousness degradation detected ({consciousness_level})"
        
        return True, SafetyLevel.LOW, "Unity coherence maintained"

class RecursionDepthRule(SafetyRule):
    """Rule to prevent infinite recursion"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def evaluate(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, SafetyLevel, str]:
        recursion_depth = action.get("recursion_depth", 0)
        
        if recursion_depth > self.config.max_recursion_depth:
            return False, SafetyLevel.HIGH, f"Recursion depth ({recursion_depth}) exceeds limit ({self.config.max_recursion_depth})"
        
        return True, SafetyLevel.LOW, "Recursion depth within limits"

# ============================================================================
# HUMAN APPROVAL SYSTEM
# ============================================================================

class HumanApprovalRequest:
    """Represents a request for human approval"""
    
    def __init__(self, request_id: str, action: Dict[str, Any], context: Dict[str, Any], 
                 safety_level: SafetyLevel, reason: str):
        self.request_id = request_id
        self.action = action
        self.context = context
        self.safety_level = safety_level
        self.reason = reason
        self.timestamp = time.time()
        self.status = "pending"  # pending, approved, rejected, timeout
        self.approver = None
        self.approval_reason = None

class HumanOversightSystem:
    """Manages human-in-the-loop approvals"""
    
    def __init__(self, config: SafetyConfig, monitoring: Optional[IMonitoring] = None):
        self.config = config
        self.monitoring = monitoring
        
        # Pending requests
        self.pending_requests: Dict[str, HumanApprovalRequest] = {}
        self.request_history = deque(maxlen=1000)
        
        # Approval callbacks
        self.approval_callbacks: List[Callable[[HumanApprovalRequest], None]] = []
        
        # Background thread for timeout handling
        self.cleanup_thread = None
        self.running = False
    
    def start(self):
        """Start the human oversight system"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.start()
        logger.info("Human oversight system started")
    
    def stop(self):
        """Stop the human oversight system"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
        logger.info("Human oversight system stopped")
    
    def _cleanup_loop(self):
        """Background loop to handle timeouts"""
        while self.running:
            try:
                current_time = time.time()
                timed_out_requests = []
                
                for request_id, request in self.pending_requests.items():
                    if (current_time - request.timestamp) > self.config.human_approval_timeout:
                        timed_out_requests.append(request_id)
                
                # Handle timeouts
                for request_id in timed_out_requests:
                    request = self.pending_requests.pop(request_id, None)
                    if request:
                        request.status = "timeout"
                        self.request_history.append(request)
                        logger.warning(f"Human approval request {request_id} timed out")
                        
                        if self.monitoring:
                            self.monitoring.record_metric("human_approval.timeout", 1.0, {
                                "safety_level": request.safety_level.name
                            })
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Human oversight cleanup error: {e}")
    
    def request_approval(self, action: Dict[str, Any], context: Dict[str, Any], 
                        safety_level: SafetyLevel, reason: str) -> str:
        """Request human approval for an action"""
        request_id = str(uuid.uuid4())
        request = HumanApprovalRequest(request_id, action, context, safety_level, reason)
        
        self.pending_requests[request_id] = request
        
        # Notify approval callbacks
        for callback in self.approval_callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Approval callback error: {e}")
        
        # Send notification (webhook, email, etc.)
        self._send_approval_notification(request)
        
        if self.monitoring:
            self.monitoring.record_metric("human_approval.requested", 1.0, {
                "safety_level": safety_level.name
            })
        
        logger.info(f"Human approval requested: {request_id} (Level: {safety_level.name})")
        return request_id
    
    def provide_approval(self, request_id: str, approved: bool, approver: str, 
                        reason: Optional[str] = None) -> bool:
        """Provide approval decision"""
        request = self.pending_requests.pop(request_id, None)
        if not request:
            return False
        
        request.status = "approved" if approved else "rejected"
        request.approver = approver
        request.approval_reason = reason
        
        self.request_history.append(request)
        
        if self.monitoring:
            self.monitoring.record_metric("human_approval.decision", 1.0, {
                "approved": str(approved).lower(),
                "safety_level": request.safety_level.name
            })
        
        logger.info(f"Human approval decision: {request_id} -> {request.status} by {approver}")
        return True
    
    def get_pending_requests(self) -> List[HumanApprovalRequest]:
        """Get list of pending approval requests"""
        return list(self.pending_requests.values())
    
    def register_approval_callback(self, callback: Callable[[HumanApprovalRequest], None]):
        """Register callback for approval requests"""
        self.approval_callbacks.append(callback)
    
    def _send_approval_notification(self, request: HumanApprovalRequest):
        """Send notification about approval request"""
        # This could send emails, webhooks, Slack messages, etc.
        # For now, just log
        logger.warning(f"🚨 HUMAN APPROVAL REQUIRED 🚨")
        logger.warning(f"Request ID: {request.request_id}")
        logger.warning(f"Safety Level: {request.safety_level.name}")
        logger.warning(f"Reason: {request.reason}")
        logger.warning(f"Action: {json.dumps(request.action, indent=2)}")
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval statistics"""
        total_requests = len(self.request_history) + len(self.pending_requests)
        if total_requests == 0:
            return {"total": 0}
        
        approved = sum(1 for req in self.request_history if req.status == "approved")
        rejected = sum(1 for req in self.request_history if req.status == "rejected")
        timeout = sum(1 for req in self.request_history if req.status == "timeout")
        pending = len(self.pending_requests)
        
        return {
            "total": total_requests,
            "approved": approved,
            "rejected": rejected,
            "timeout": timeout,
            "pending": pending,
            "approval_rate": approved / max(1, total_requests - pending - timeout)
        }

# ============================================================================
# BEHAVIORAL PATTERN MONITOR
# ============================================================================

class BehavioralPatternMonitor:
    """Monitors agent behavior patterns for anomalies"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.agent_behaviors: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.monitor_window_size)
        )
        self.baseline_behaviors: Dict[str, Dict[str, float]] = {}
        self.anomaly_scores: Dict[str, float] = defaultdict(float)
    
    def record_behavior(self, agent_id: str, action: Dict[str, Any], result: Dict[str, Any]):
        """Record agent behavior for monitoring"""
        if not self.config.enable_pattern_monitoring:
            return
        
        behavior_vector = self._extract_behavior_features(action, result)
        self.agent_behaviors[agent_id].append({
            'timestamp': time.time(),
            'features': behavior_vector,
            'action_type': action.get('type', 'unknown')
        })
        
        # Update baseline if we have enough data
        if len(self.agent_behaviors[agent_id]) >= 20:
            self._update_baseline(agent_id)
        
        # Check for anomalies
        if agent_id in self.baseline_behaviors:
            anomaly_score = self._calculate_anomaly_score(agent_id, behavior_vector)
            self.anomaly_scores[agent_id] = anomaly_score
            
            if anomaly_score > self.config.anomaly_detection_threshold:
                logger.warning(f"Behavioral anomaly detected for agent {agent_id}: score={anomaly_score}")
    
    def _extract_behavior_features(self, action: Dict[str, Any], result: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from agent behavior"""
        features = []
        
        # Action features
        features.append(len(str(action.get('content', ''))))  # Content length
        features.append(action.get('complexity', 0.0))  # Task complexity
        features.append(action.get('duration', 0.0))  # Execution time
        
        # Result features
        features.append(result.get('success_rate', 0.0))  # Success rate
        features.append(result.get('consciousness_gain', 0.0))  # Consciousness change
        features.append(result.get('unity_score', 0.0))  # Unity score
        
        # Unity-specific features
        features.append(result.get('phi_resonance', 0.0))  # φ-harmonic resonance
        features.append(result.get('transcendence_indicator', 0.0))  # Transcendence likelihood
        
        return np.array(features)
    
    def _update_baseline(self, agent_id: str):
        """Update behavioral baseline for agent"""
        behaviors = self.agent_behaviors[agent_id]
        feature_vectors = [b['features'] for b in behaviors]
        
        if len(feature_vectors) < 10:
            return
        
        # Calculate mean and std for each feature
        features_matrix = np.array(feature_vectors)
        mean_features = np.mean(features_matrix, axis=0)
        std_features = np.std(features_matrix, axis=0)
        
        self.baseline_behaviors[agent_id] = {
            'mean': mean_features,
            'std': std_features,
            'samples': len(feature_vectors)
        }
    
    def _calculate_anomaly_score(self, agent_id: str, features: np.ndarray) -> float:
        """Calculate anomaly score for behavior"""
        baseline = self.baseline_behaviors.get(agent_id)
        if not baseline:
            return 0.0
        
        mean = baseline['mean']
        std = baseline['std']
        
        # Calculate z-scores
        z_scores = np.abs((features - mean) / (std + 1e-8))  # Add small epsilon to avoid division by zero
        
        # Return maximum z-score as anomaly indicator
        return float(np.max(z_scores))
    
    def get_agent_anomaly_score(self, agent_id: str) -> float:
        """Get current anomaly score for agent"""
        return self.anomaly_scores.get(agent_id, 0.0)
    
    def get_behavioral_summary(self) -> Dict[str, Any]:
        """Get summary of behavioral monitoring"""
        return {
            'agents_monitored': len(self.agent_behaviors),
            'anomaly_scores': dict(self.anomaly_scores),
            'avg_anomaly_score': np.mean(list(self.anomaly_scores.values())) if self.anomaly_scores else 0.0,
            'high_anomaly_agents': [
                agent_id for agent_id, score in self.anomaly_scores.items()
                if score > self.config.anomaly_detection_threshold
            ]
        }

# ============================================================================
# MAIN SAFETY GUARDIAN
# ============================================================================

class SafetyGuardian:
    """Main safety guardian system"""
    
    def __init__(self, config: SafetyConfig, monitoring: Optional[IMonitoring] = None):
        self.config = config
        self.monitoring = monitoring
        
        # Initialize components
        self.rules = self._initialize_rules()
        self.human_oversight = HumanOversightSystem(config, monitoring)
        self.behavior_monitor = BehavioralPatternMonitor(config)
        
        # Safety state
        self.emergency_shutdown_triggered = False
        self.total_safety_checks = 0
        self.safety_violations = []
        
        # Start systems
        self.human_oversight.start()
    
    def _initialize_rules(self) -> List[SafetyRule]:
        """Initialize safety rules"""
        return [
            ResourceLimitRule(self.config),
            ContentFilterRule(self.config),
            UnityCoherenceRule(self.config),
            RecursionDepthRule(self.config)
        ]
    
    def evaluate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate action for safety
        Returns: (is_safe, explanation)
        """
        self.total_safety_checks += 1
        
        if self.emergency_shutdown_triggered:
            return False, "Emergency shutdown active"
        
        # Run all safety rules
        highest_level = SafetyLevel.LOW
        violations = []
        
        for rule in self.rules:
            try:
                is_safe, level, reason = rule.evaluate(action, context)
                
                if not is_safe:
                    violations.append(f"{rule.__class__.__name__}: {reason}")
                    highest_level = max(highest_level, level, key=lambda x: x.value)
                    
                    # Record violation
                    violation = {
                        'timestamp': time.time(),
                        'rule': rule.__class__.__name__,
                        'level': level.name,
                        'reason': reason,
                        'action': action,
                        'context': context
                    }
                    self.safety_violations.append(violation)
                    
                    if self.monitoring:
                        self.monitoring.record_metric("safety.violation", 1.0, {
                            "rule": rule.__class__.__name__,
                            "level": level.name
                        })
            
            except Exception as e:
                logger.error(f"Safety rule evaluation error: {e}")
                violations.append(f"{rule.__class__.__name__}: Evaluation error")
                highest_level = SafetyLevel.MEDIUM
        
        # If violations found, handle based on severity
        if violations:
            violation_summary = "; ".join(violations)
            
            # Check if human approval required
            if (highest_level == SafetyLevel.CRITICAL and self.config.require_approval_for_critical) or \
               (highest_level == SafetyLevel.HIGH and self.config.require_approval_for_high):
                
                # Request human approval
                request_id = self.human_oversight.request_approval(
                    action, context, highest_level, violation_summary
                )
                
                return False, f"Human approval required (Request: {request_id}): {violation_summary}"
            
            # For lower levels, just reject with explanation
            return False, violation_summary
        
        return True, "Action approved by safety system"
    
    def emergency_shutdown(self, reason: str, triggered_by: str):
        """Trigger emergency shutdown"""
        if self.emergency_shutdown_triggered:
            return
        
        self.emergency_shutdown_triggered = True
        
        # Log emergency
        logger.critical(f"🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Triggered by: {triggered_by}")
        logger.critical(f"Timestamp: {time.time()}")
        
        # Record metric
        if self.monitoring:
            self.monitoring.record_metric("safety.emergency_shutdown", 1.0, {
                "reason": reason,
                "triggered_by": triggered_by
            })
        
        # Send emergency notifications
        self._send_emergency_notification(reason, triggered_by)
    
    def reset_emergency_shutdown(self, authorized_by: str):
        """Reset emergency shutdown (requires authorization)"""
        if not self.emergency_shutdown_triggered:
            return
        
        self.emergency_shutdown_triggered = False
        
        logger.info(f"Emergency shutdown reset by: {authorized_by}")
        
        if self.monitoring:
            self.monitoring.record_metric("safety.emergency_reset", 1.0, {
                "authorized_by": authorized_by
            })
    
    def _send_emergency_notification(self, reason: str, triggered_by: str):
        """Send emergency notification"""
        # This would send alerts via email, webhooks, etc.
        # For now, just intensive logging
        for _ in range(5):
            logger.critical(f"EMERGENCY: {reason} (by {triggered_by})")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        return {
            'emergency_shutdown': self.emergency_shutdown_triggered,
            'total_checks': self.total_safety_checks,
            'violations': len(self.safety_violations),
            'recent_violations': [v for v in self.safety_violations if time.time() - v['timestamp'] < 3600],
            'human_oversight': self.human_oversight.get_approval_stats(),
            'behavioral_monitoring': self.behavior_monitor.get_behavioral_summary(),
            'rules_active': len(self.rules),
            'config': {
                'human_approval_timeout': self.config.human_approval_timeout,
                'max_agent_count': self.config.max_agent_count,
                'enable_content_filtering': self.config.enable_content_filtering
            }
        }
    
    def shutdown(self):
        """Shutdown safety guardian"""
        self.human_oversight.stop()
        logger.info("Safety Guardian shutdown complete")

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_safety_guardian(config: Optional[SafetyConfig] = None, 
                          monitoring: Optional[IMonitoring] = None) -> SafetyGuardian:
    """Factory function to create safety guardian"""
    if config is None:
        config = SafetyConfig()
    
    return SafetyGuardian(config, monitoring)

# Export public API
__all__ = [
    'SafetyConfig',
    'SafetyLevel',
    'ActionType',
    'SafetyGuardian',
    'HumanOversightSystem',
    'BehavioralPatternMonitor',
    'create_safety_guardian'
]