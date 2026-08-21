# LLM abstraction layer

from __future__ import annotations

from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Concrete implementations should provide a ``generate_text`` method that
    accepts a prompt string and returns a string response.
    """

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's textual response.
        """
        raise NotImplementedError

# Example dummy provider for testing purposes
class DummyLLMProvider(LLMProvider):
    def generate_text(self, prompt: str) -> str:  # pragma: no cover
        return f"[Dummy response to: {prompt}]"
