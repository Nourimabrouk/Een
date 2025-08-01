from typing import Any

from .base import BaseAgent


class EchoAgent(BaseAgent):
    """A toy agent that simply echoes the provided message.

    Example invocation:
        spawn-agent src.agents.echo_agent:EchoAgent --kwargs message="Hello"
    """

    def run(self) -> Any:
        message: str = self.config.get("message", "Hello, World!")
        return {"echo": message}