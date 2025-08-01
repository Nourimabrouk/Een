from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Lightweight abstract base class for all agents.

    Agents should implement :py:meth:`run` which carries out their primary task and returns a
    serialisable result (e.g. dict, str, list).
    """

    def __init__(self, **kwargs):
        """Accept arbitrary keyword arguments for downstream flexibility."""
        # Store config in case subclasses need access later.
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def run(self) -> Any:
        """Execute the agent and return a result."""

    def __call__(self, *args, **kwargs):
        # Allow instances to be directly callable.
        return self.run()