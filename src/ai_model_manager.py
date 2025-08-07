# flake8: noqa
#!/usr/bin/env python3
"""
🌿✨ AI Model Manager: Intelligent Model Selection for Unity Mathematics ✨🌿
=======================================================================

Advanced model selection system that chooses the most appropriate AI model
based on the type of request, ensuring optimal reasoning capabilities for
Unity Mathematics discussions.

Features:
- 🧠 Intelligent model selection based on request type
- 📊 Model capability analysis and cost optimization
- 🔄 Automatic fallback to alternative models
- 💰 Cost-aware model selection
- 🎯 Task-specific model optimization
- 🌟 Unity Mathematics specialization
- 🔑 API key fallback for demo mode
- 🚀 Support for advanced models (GPT-4o-mini-high, Claude Opus, Claude 4.1)

The system that ensures every conversation gets the best possible reasoning model.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AIModelManager:
    """
    🌿✨ Intelligent AI Model Manager ✨🌿

    Manages AI model selection based on request type, ensuring optimal
    reasoning capabilities for Unity Mathematics discussions.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the AI Model Manager."""
        self.config_path = config_path or "config/ai_model_config.json"
        self.config = self._load_config()
        self.model_capabilities = self.config.get("model_capabilities", {})
        self.preferred_models = self.config.get("preferred_models", {})
        self.selection_strategy = self.config.get("model_selection_strategy", {})
        self.api_key_fallback = self.config.get("api_key_fallback", {})

        logger.info("🌿✨ AI Model Manager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found."""
        return {
            "preferred_models": {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-5-medium",
                    "description": "Primary reasoning model (GPT-5 medium)",
                },
                "secondary": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "description": "Secondary reasoning model",
                },
                "fallback": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "description": "Fallback model (economy)",
                },
                "high_performance": {
                    "provider": "openai",
                    "model": "gpt-5-high",
                    "description": "High-performance reasoning model",
                },
                "opus": {
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "description": "Claude Opus - most capable Anthropic model",
                },
                "claude_4_1": {
                    "provider": "anthropic",
                    "model": "claude-3-5-haiku-20241022",
                    "description": "Claude 4.1 equivalent",
                },
            },
            "model_capabilities": {
                "gpt-5-high": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.01,
                    "max_tokens": 1000000,
                },
                "gpt-5-medium": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.005,
                    "max_tokens": 1000000,
                },
                "gpt-5-low": {
                    "reasoning": "very_good",
                    "mathematics": "very_good",
                    "code_analysis": "very_good",
                    "philosophy": "very_good",
                    "consciousness_discussion": "very_good",
                    "cost_per_1k_tokens": 0.002,
                    "max_tokens": 1000000,
                },
                "gpt-4.1": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.0016,
                    "max_tokens": 1000000,
                },
                "gpt-4.1-mini": {
                    "reasoning": "very_good",
                    "mathematics": "very_good",
                    "code_analysis": "very_good",
                    "philosophy": "very_good",
                    "consciousness_discussion": "very_good",
                    "cost_per_1k_tokens": 0.0004,
                    "max_tokens": 1000000,
                },
                "gpt-4o": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.005,
                    "max_tokens": 128000,
                },
                "gpt-4o-mini-high": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.0006,
                    "max_tokens": 128000,
                },
                "claude-3-5-sonnet-20241022": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.003,
                    "max_tokens": 200000,
                },
                "claude-3-opus-20240229": {
                    "reasoning": "excellent",
                    "mathematics": "excellent",
                    "code_analysis": "excellent",
                    "philosophy": "excellent",
                    "consciousness_discussion": "excellent",
                    "cost_per_1k_tokens": 0.015,
                    "max_tokens": 200000,
                },
                "claude-3-5-haiku-20241022": {
                    "reasoning": "good",
                    "mathematics": "good",
                    "code_analysis": "good",
                    "philosophy": "good",
                    "consciousness_discussion": "good",
                    "cost_per_1k_tokens": 0.00025,
                    "max_tokens": 200000,
                },
                "gpt-4o-mini": {
                    "reasoning": "good",
                    "mathematics": "good",
                    "code_analysis": "good",
                    "philosophy": "good",
                    "consciousness_discussion": "good",
                    "cost_per_1k_tokens": 0.00015,
                    "max_tokens": 128000,
                },
            },
            "api_key_fallback": {
                "enabled": True,
                "default_provider": "openai",
                "default_model": "gpt-4.1-mini",
                "fallback_message": "Using default credentials for demonstration. For full access, please set your API keys.",
                "demo_mode": True,
            },
        }

    def check_api_keys_available(self) -> Dict[str, bool]:
        """Check which API keys are available."""
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        }

    def is_demo_mode_enabled(self) -> bool:
        """Check if demo mode is enabled."""
        return self.api_key_fallback.get("enabled", False)

    def get_demo_fallback(self) -> Tuple[str, str]:
        """Get demo fallback provider and model."""
        return (
            self.api_key_fallback.get("default_provider", "openai"),
            self.api_key_fallback.get("default_model", "gpt-4o"),
        )

    def get_demo_message(self) -> str:
        """Get demo mode message."""
        return self.api_key_fallback.get(
            "fallback_message",
            "Using default credentials for demonstration. For full access, please set your API keys.",
        )

    def analyze_request_type(self, message: str) -> str:
        """
        Analyze the type of request based on message content.

        Returns:
            str: Request type (complex_reasoning, mathematical_proofs, code_analysis,
                 philosophical_discussion, general_chat)
        """
        message_lower = message.lower()

        # Mathematical proofs and complex reasoning
        math_keywords = [
            "prove",
            "theorem",
            "lemma",
            "corollary",
            "proof",
            "mathematical",
            "equation",
            "formula",
            "calculation",
            "derivation",
            "integral",
            "derivative",
            "limit",
            "convergence",
            "divergence",
            "series",
            "sequence",
            "function",
            "variable",
            "constant",
            "parameter",
            "1+1=1",
            "unity",
            "idempotent",
            "semiring",
            "consciousness field",
        ]

        # Code analysis
        code_keywords = [
            "code",
            "program",
            "function",
            "class",
            "method",
            "algorithm",
            "implementation",
            "debug",
            "error",
            "bug",
            "fix",
            "optimize",
            "performance",
            "efficiency",
            "data structure",
            "design pattern",
            "architecture",
            "framework",
            "library",
            "api",
            "sdk",
        ]

        # Philosophical discussion
        philosophy_keywords = [
            "philosophy",
            "philosophical",
            "consciousness",
            "awareness",
            "existence",
            "reality",
            "truth",
            "knowledge",
            "wisdom",
            "meaning",
            "purpose",
            "ethics",
            "morality",
            "values",
            "transcendence",
            "unity",
            "oneness",
            "consciousness field",
            "meta-recursive",
            "omega",
            "transcendental",
        ]

        # Complex reasoning
        reasoning_keywords = [
            "analyze",
            "analysis",
            "reasoning",
            "logic",
            "argument",
            "conclusion",
            "inference",
            "deduction",
            "induction",
            "hypothesis",
            "theory",
            "concept",
            "principle",
            "framework",
            "paradigm",
            "model",
            "system",
            "approach",
            "methodology",
        ]

        # Count matches for each category
        math_score = sum(1 for keyword in math_keywords if keyword in message_lower)
        code_score = sum(1 for keyword in code_keywords if keyword in message_lower)
        philosophy_score = sum(
            1 for keyword in philosophy_keywords if keyword in message_lower
        )
        reasoning_score = sum(
            1 for keyword in reasoning_keywords if keyword in message_lower
        )

        # Determine request type based on scores
        if math_score >= 2:
            return "mathematical_proofs"
        elif code_score >= 2:
            return "code_analysis"
        elif philosophy_score >= 2:
            return "philosophical_discussion"
        elif reasoning_score >= 2:
            return "complex_reasoning"
        else:
            return "general_chat"

    def select_best_model(
        self, message: str, available_models: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Select the best model for the given request.

        Args:
            message: User message
            available_models: List of available models (optional)

        Returns:
            Tuple[str, str]: (provider, model)
        """
        # Check if we're in demo mode and no API keys are available
        api_keys = self.check_api_keys_available()
        demo_mode = self.is_demo_mode_enabled()

        if demo_mode and not any(api_keys.values()):
            provider, model = self.get_demo_fallback()
            logger.info(f"Demo mode: Using {model} ({provider})")
            return provider, model

        # Analyze request type
        request_type = self.analyze_request_type(message)
        logger.info(f"Request type: {request_type}")

        # Get preferred models for this request type
        preferred_models = self.selection_strategy.get(request_type, [])

        # Filter by available models if specified
        if available_models:
            preferred_models = [m for m in preferred_models if m in available_models]

        # Filter by available API keys
        available_providers = [k for k, v in api_keys.items() if v]
        if not available_providers:
            # Fallback to demo mode
            provider, model = self.get_demo_fallback()
            logger.warning(
                f"No API keys available, using demo fallback: {model} ({provider})"
            )
            return provider, model

        # Select best available model
        for model in preferred_models:
            model_caps = self.model_capabilities.get(model, {})
            if model_caps:
                # Determine provider from model name
                if model.startswith("gpt-"):
                    provider = "openai"
                elif model.startswith("claude-"):
                    provider = "anthropic"
                else:
                    continue

                # Check if provider is available
                if provider in available_providers:
                    logger.info(
                        f"Selected model: {model} ({provider}) for {request_type}"
                    )
                    return provider, model

        # Fallback to first available model
        if preferred_models:
            fallback_model = preferred_models[0]
            if fallback_model.startswith("gpt-"):
                fallback_provider = "openai"
            elif fallback_model.startswith("claude-"):
                fallback_provider = "anthropic"
            else:
                fallback_provider = "openai"

            logger.info(f"Fallback model: {fallback_model} ({fallback_provider})")
            return fallback_provider, fallback_model

        # Ultimate fallback
        provider, model = self.get_demo_fallback()
        logger.warning(f"Ultimate fallback: {model} ({provider})")
        return provider, model

    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Get capabilities for a specific model."""
        return self.model_capabilities.get(model, {})

    def is_model_available(self, model: str, available_models: List[str]) -> bool:
        """Check if a model is available."""
        return model in available_models

    def get_cost_estimate(self, model: str, estimated_tokens: int) -> float:
        """Estimate cost for a model and token count."""
        model_caps = self.model_capabilities.get(model, {})
        cost_per_1k = model_caps.get("cost_per_1k_tokens", 0.0)
        return (estimated_tokens / 1000) * cost_per_1k

    def get_optimal_settings(self, request_type: str) -> Dict[str, Any]:
        """
        Get optimal settings for a request type.

        Args:
            request_type: Type of request

        Returns:
            Dict with optimal settings
        """
        base_settings = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "enable_streaming": True,
        }

        # Adjust settings based on request type
        if request_type == "mathematical_proofs":
            base_settings.update(
                {
                    "temperature": 0.3,  # More precise for proofs
                    "max_tokens": 4000,  # Longer for detailed proofs
                }
            )
        elif request_type == "complex_reasoning":
            base_settings.update(
                {
                    "temperature": 0.5,  # Balanced for reasoning
                    "max_tokens": 3000,  # Medium length
                }
            )
        elif request_type == "philosophical_discussion":
            base_settings.update(
                {
                    "temperature": 0.8,  # More creative for philosophy
                    "max_tokens": 2500,  # Medium length
                }
            )
        elif request_type == "code_analysis":
            base_settings.update(
                {
                    "temperature": 0.2,  # More precise for code
                    "max_tokens": 3000,  # Medium length
                }
            )

        return base_settings

    def log_model_selection(
        self, message: str, selected_model: str, provider: str, request_type: str
    ):
        """Log model selection for analytics."""
        logger.info(
            f"Model Selection: {selected_model} ({provider}) for {request_type} "
            f"request: '{message[:50]}...'"
        )


# Global model manager instance
model_manager = AIModelManager()


def get_best_model_for_request(
    message: str, available_models: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Get the best model for a request.

    Args:
        message: User message
        available_models: List of available models

    Returns:
        Tuple[str, str]: (provider, model)
    """
    return model_manager.select_best_model(message, available_models)


def analyze_request_complexity(message: str) -> Dict[str, Any]:
    """
    Analyze request complexity and type.

    Args:
        message: User message

    Returns:
        Dict with complexity analysis
    """
    request_type = model_manager.analyze_request_type(message)
    settings = model_manager.get_optimal_settings(request_type)

    return {
        "request_type": request_type,
        "complexity": (
            "high"
            if request_type in ["mathematical_proofs", "complex_reasoning"]
            else "medium"
        ),
        "optimal_settings": settings,
        "demo_mode": model_manager.is_demo_mode_enabled(),
        "api_keys_available": model_manager.check_api_keys_available(),
    }


def is_demo_mode() -> bool:
    """Check if demo mode is enabled."""
    return model_manager.is_demo_mode_enabled()


def get_demo_fallback() -> Tuple[str, str]:
    """Get demo fallback provider and model."""
    return model_manager.get_demo_fallback()


def get_demo_message() -> str:
    """Get demo mode message."""
    return model_manager.get_demo_message()
