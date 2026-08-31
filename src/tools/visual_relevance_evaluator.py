# Vision-capable relevance evaluation abstraction for Visual QC.
#
# Kept as its own interface (not folded into LLMProvider): LLMProvider's
# generate_text(prompt) -> str contract has no way to carry images, and
# extending it would mean every existing caller (Research/Script agents)
# gains a parameter they never use. VisualQCService depends only on this
# interface, never on a concrete vision model directly.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.models.visual_qc import RawAssetVerdict


class VisualRelevanceEvaluatorError(Exception):
    """Raised when a vision evaluation call fails outright (network/API
    error, unparseable response). Caught by VisualQCService, which falls
    back to metadata-only approval for the affected section rather than
    failing QC entirely - see VisualQCService's fallback handling."""


@dataclass
class AssetFrames:
    """One candidate asset's representative frame(s), ready to submit for
    evaluation. For an image asset, this is just the image file itself."""

    asset_id: str
    frame_paths: List[str]


@dataclass
class SectionQCContext:
    """Everything one evaluate_section() call needs for one script
    section's batch of candidate assets - concise by design (see
    VisualRelevanceEvaluator docstring): no full research context, no
    entire video files, just what's needed to judge these specific frames."""

    topic: str
    section_index: int
    section_heading: str
    narration: str
    semantic_summary: str
    visual_intents: List[str]
    avoid_concepts: List[str]
    assets: List[AssetFrames] = field(default_factory=list)


class VisualRelevanceEvaluator(ABC):
    """Abstract interface for judging whether representative frames from a
    selected media asset actually fit a script section's meaning.

    One call is expected per script section, batching every asset selected
    for that section's visual slots - never one call per frame, and never
    one call per slot - to keep vision-model usage cheap regardless of how
    many slots a section needs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short evaluator identifier, e.g. 'mock', 'gemini'."""
        raise NotImplementedError

    @abstractmethod
    async def evaluate_section(self, context: SectionQCContext) -> List[RawAssetVerdict]:
        """Evaluate every asset in ``context.assets`` in a single request.

        Args:
            context: Section meaning/intent plus each candidate asset's
                representative frame(s)

        Returns:
            One RawAssetVerdict per asset in ``context.assets`` (any
            asset_id present in the input should have a corresponding
            verdict in the output)

        Raises:
            VisualRelevanceEvaluatorError: On a call failure or an
                unusable/unparseable response
        """
        raise NotImplementedError


class MockVisualRelevanceEvaluator(VisualRelevanceEvaluator):
    """Mock evaluator for development/testing without a real vision model.

    Returns a configurable, deterministic verdict per asset_id (or a
    default verdict for any asset_id not explicitly configured), and
    records every call it received for test assertions.
    """

    def __init__(
        self,
        verdicts_by_asset_id: Optional[Dict[str, RawAssetVerdict]] = None,
        default_score: float = 0.8,
        raise_error: Optional[Exception] = None,
    ) -> None:
        """Initialize the mock evaluator.

        Args:
            verdicts_by_asset_id: Explicit verdict to return for specific
                asset ids, for testing approval/rejection scenarios
            default_score: relevance_score used for any asset_id not in
                verdicts_by_asset_id
            raise_error: If set, evaluate_section always raises this
                instead of returning verdicts, to simulate a provider outage
        """
        self.verdicts_by_asset_id = verdicts_by_asset_id or {}
        self.default_score = default_score
        self.raise_error = raise_error
        self.calls: List[SectionQCContext] = []

    @property
    def name(self) -> str:
        return "mock"

    async def evaluate_section(self, context: SectionQCContext) -> List[RawAssetVerdict]:
        self.calls.append(context)
        if self.raise_error is not None:
            raise self.raise_error

        verdicts = []
        for asset in context.assets:
            verdict = self.verdicts_by_asset_id.get(asset.asset_id)
            if verdict is None:
                verdict = RawAssetVerdict(
                    asset_id=asset.asset_id, relevance_score=self.default_score, reason="mock default"
                )
            verdicts.append(verdict)
        return verdicts
