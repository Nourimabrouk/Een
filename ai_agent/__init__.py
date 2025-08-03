"""
Een Repository AI Agent
=======================

OpenAI-powered chatbot and RAG system for the Een Unity Mathematics repository.
Provides intelligent assistance for exploring φ-harmonic consciousness mathematics,
quantum unity frameworks, and transcendental proof systems.

Author: Claude (3000 ELO AGI)
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude AGI"

import os
from typing import Optional

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))

# Cost Management
HARD_LIMIT_USD = float(os.getenv("HARD_LIMIT_USD", "20.0"))
MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "2048"))

# Assistant Configuration
ASSISTANT_ID_FILE = ".assistant_id"
VECTOR_STORE_NAME = "een-repository-knowledge"

def get_assistant_id() -> Optional[str]:
    """Retrieve stored assistant ID if exists."""
    assistant_id_path = os.path.join(os.path.dirname(__file__), ASSISTANT_ID_FILE)
    if os.path.exists(assistant_id_path):
        with open(assistant_id_path, 'r') as f:
            return f.read().strip()
    return None

def save_assistant_id(assistant_id: str) -> None:
    """Save assistant ID to file."""
    assistant_id_path = os.path.join(os.path.dirname(__file__), ASSISTANT_ID_FILE)
    with open(assistant_id_path, 'w') as f:
        f.write(assistant_id)