# Mock LLM provider for development/testing
from __future__ import annotations

from typing import Any

from src.llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns realistic responses for testing.

    Avoids the need for API keys during development and testing.
    """

    def __init__(self, response_map: dict[str, str] | None = None) -> None:
        """Initialize with optional custom response mapping.

        Args:
            response_map: Dictionary mapping input patterns to responses
        """
        self.response_map = response_map or {}

    def generate_text(self, prompt: str) -> str:
        """Generate a mock response based on the prompt.

        Uses pattern matching for common research tasks.
        """
        prompt_lower = prompt.lower()

        # Custom responses from mapping
        for pattern, response in self.response_map.items():
            if pattern.lower() in prompt_lower:
                return response

        # Default responses for common research patterns
        if "summarize" in prompt_lower:
            return (
                "Dreams occur primarily during REM sleep and serve multiple functions "
                "including memory consolidation, emotional processing, and learning. "
                "Brain imaging shows increased activity in the visual, motor, and "
                "emotional regions during dreaming, while the prefrontal cortex shows "
                "decreased activity, explaining the illogical nature of dreams."
            )
        elif "extract" in prompt_lower and "fact" in prompt_lower:
            return (
                "1. Dreams primarily occur during REM (Rapid Eye Movement) sleep\n"
                "2. The brain consolidates memories and processes emotions during dreaming\n"
                "3. Most vivid dreams happen in REM cycles approximately every 90 minutes\n"
                "4. Prefrontal cortex activity decreases during dreams, explaining illogical narratives\n"
                "5. Average person spends about 2 hours dreaming each night\n"
                "6. Dreams may serve as threat simulation and problem-solving practice\n"
                "7. External stimuli can be incorporated into dreams (dream incorporation)\n"
                "8. Lucid dreaming occurs when dreamers become aware they are dreaming\n"
                "9. Dream recall varies significantly between individuals\n"
                "10. Certain medications and substances can affect dream frequency and intensity"
            )
        elif "organize" in prompt_lower or "structure" in prompt_lower:
            return (
                "The research on dreaming reveals several key themes: "
                "1) Biological mechanisms of REM sleep and brain activity patterns, "
                "2) Psychological functions including memory consolidation and emotional regulation, "
                "3) Evolutionary advantages such as threat simulation and problem-solving, "
                "4) Individual differences in dream recall and content, "
                "5) Practical applications in therapy and creativity enhancement."
            )
        elif "key point" in prompt_lower:
            return (
                "• Dreams occur mainly during REM sleep cycles\n"
                "• Brain consolidates memories and processes emotions while dreaming\n"
                "• Most adults spend about 2 hours per night dreaming\n"
                "• Prefrontal cortex suppression creates dream illogic\n"
                "• Dreams may serve evolutionary functions like threat simulation"
            )
        else:
            # Generic fallback response
            return (
                f"Based on research into '{prompt[:50]}...', here are the key findings: "
                "The topic involves multiple interconnected aspects that have been studied "
                "extensively. Current understanding suggests complex interactions between "
                "biological, psychological, and environmental factors. Research indicates "
                "several important implications for both theoretical understanding and "
                "practical applications in related fields."
            )