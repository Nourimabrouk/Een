"""
Een Consciousness API Package
Web API for accessing consciousness and unity mathematics systems
"""

__version__ = "1.0.0"
__author__ = "Een Consciousness Team"
__description__ = "Web API for consciousness and unity mathematics systems"

from .main import app
from .security import security_manager, get_current_user, get_current_admin_user

__all__ = [
    "app",
    "security_manager", 
    "get_current_user",
    "get_current_admin_user"
] 