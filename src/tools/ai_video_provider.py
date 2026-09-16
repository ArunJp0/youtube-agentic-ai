# AI Video Generation provider abstraction. Mirrors every other provider
# interface in this project (MediaProvider, VoiceProvider, LLMProvider,
# TopicSourceProvider, ...): the application (VisualMediaService/
# VisualContextPlanner) depends only on this interface, never on a
# concrete vendor SDK (fal.ai, Kling, Seedance, Veo, PixVerse, ...)
# directly - a real implementation for any of those is a future,
# separate module, added purely by implementing this same ABC.
#
# generate() takes a fully-formed AIVideoGenerationRequest (built upstream
# by VisualMediaService from the Visual Context Planner's own plan, never
# by this module) and returns an AIVideoGenerationResult - never raises for
# a generation failure, exactly like YouTubeClient/MediaProvider/
# LLMProvider's own real implementations.
from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.ai_video import AIVideoGenerationRequest, AIVideoGenerationResult


class AIVideoProviderError(Exception):
    """Raised only for configuration/programmer errors (e.g. a
    LocalAIVideoProvider with no clips directory configured). A real
    generation failure (API error, timeout, content rejected, quota
    exceeded) is NEVER raised - it's captured in the returned
    AIVideoGenerationResult so callers always get a structured result back."""


class AIVideoProvider(ABC):
    """Abstract base class for AI video clip generation.

    A concrete implementation owns all vendor-specific request/response
    translation, auth, polling/webhook handling, and retry-worthy-error
    classification - none of that is visible to callers, which only ever
    see AIVideoGenerationRequest in, AIVideoGenerationResult out.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. 'mock', 'local_ai_video', or a
        future 'fal_kling'/'pixverse'."""
        raise NotImplementedError

    @abstractmethod
    async def generate(self, request: AIVideoGenerationRequest) -> AIVideoGenerationResult:
        """Generate (or otherwise obtain) one short clip for ``request``.

        Returns:
            Structured AIVideoGenerationResult - never raises for a
            generation/API failure, only for genuine misconfiguration
            (see AIVideoProviderError).
        """
        raise NotImplementedError


class MockAIVideoProvider(AIVideoProvider):
    """In-memory test double - no network, no filesystem writes beyond a
    tiny placeholder file. Returns deterministic fake results, or a
    simulated failure, so VisualMediaService/pipeline tests can exercise
    AI-video request/response handling and bounded-retry/fallback behavior
    without any real provider."""

    def __init__(
        self,
        fail: bool = False,
        fail_times: int = 0,
        duration_seconds: float = 6.0,
        provider_name: str = "mock",
        write_placeholder_file: bool = True,
    ) -> None:
        """Initialize the mock provider.

        Args:
            fail: If True, every call returns a failed result.
            fail_times: Number of leading calls to fail before succeeding -
                for testing bounded-retry behavior deterministically
                (ignored if ``fail`` is True).
            duration_seconds: Clip duration reported on a successful result.
            provider_name: Value returned by ``name`` - lets tests
                distinguish multiple mock instances.
            write_placeholder_file: When True (default), a successful
                result's ``local_file_path`` points to a real small
                placeholder file written under a private temp directory -
                so callers that copy/move the "generated" clip (see
                VisualMediaService._materialize_ai_asset) have something
                real to operate on, exactly like MockMediaProvider.download()
                already does for stock assets. When False, the path is a
                non-existent placeholder string only.
        """
        self.fail = fail
        self.fail_times = fail_times
        self.duration_seconds = duration_seconds
        self._provider_name = provider_name
        self._temp_dir = tempfile.mkdtemp(prefix="mock-ai-video-") if write_placeholder_file else None
        self.calls: List[AIVideoGenerationRequest] = []

    @property
    def name(self) -> str:
        return self._provider_name

    async def generate(self, request: AIVideoGenerationRequest) -> AIVideoGenerationResult:
        self.calls.append(request)
        attempt = len(self.calls)

        if self.fail or attempt <= self.fail_times:
            return AIVideoGenerationResult(
                success=False,
                status="failed",
                provider=self._provider_name,
                error="simulated AI video generation failure",
            )

        if self._temp_dir:
            local_file_path = os.path.join(
                self._temp_dir, f"{request.section_index}-{request.slot_index}-{attempt}.mp4"
            )
            with open(local_file_path, "wb") as f:
                f.write(b"MOCK AI VIDEO CLIP")
        else:
            local_file_path = f"/mock/ai_video/{request.section_index}-{request.slot_index}.mp4"

        return AIVideoGenerationResult(
            success=True,
            status="succeeded",
            provider=self._provider_name,
            model="mock-model",
            local_file_path=local_file_path,
            duration_seconds=self.duration_seconds,
            width=1920,
            height=1080,
            provider_job_id=f"mock-job-{attempt}",
            cost_usd=0.0,
        )


class LocalAIVideoProvider(AIVideoProvider):
    """DEMO / LOCAL PROVIDER - NOT a production AI-generation provider.

    This does NOT call any AI video generation API and does NOT generate
    anything itself. It only serves already-generated local clip files
    (e.g. manually produced ahead of time using a reputable free web tier
    like PixVerse's Basic plan, for demonstrating visual quality to a
    client before purchasing production API credits) - matching each
    request to a pre-placed file by section/slot index, or consuming files
    from a directory in order.

    Never claims manually-generated clips were produced "autonomously by
    an API" - ``model`` is always reported as ``"manual-local-demo-clip"``,
    never a real model name, so a result can never be mistaken for a real
    API generation downstream (provenance/compliance records, etc.).
    """

    DEMO_MODEL_LABEL = "manual-local-demo-clip"

    def __init__(self, clips_dir: str, provider_name: str = "local_ai_video") -> None:
        """Initialize the provider.

        Args:
            clips_dir: Local directory containing pre-generated clip files,
                named ``<section_index>-<slot_index>.<ext>`` (e.g.
                ``0-0.mp4``) so each request maps to exactly one file.
            provider_name: Value returned by ``name``.

        Raises:
            AIVideoProviderError: If ``clips_dir`` is falsy - a
                configuration error, not a generation failure.
        """
        if not clips_dir:
            raise AIVideoProviderError("LocalAIVideoProvider requires a clips_dir")
        self.clips_dir = clips_dir
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    async def generate(self, request: AIVideoGenerationRequest) -> AIVideoGenerationResult:
        """Serve a pre-generated demo clip for this exact slot when one
        exists. With a limited free-credit clip set, an exact match won't
        always exist - rather than failing the slot (which, with stock
        fallback disabled for an AI-only demo, would otherwise leave it
        empty) or ever reaching for Pexels, this reuses another already-
        served demo clip: first another slot from the SAME section (the
        most visually coherent choice), then any clip generated for this
        request at all. Every reuse is reported via ``warnings``, never
        silently substituted."""
        warnings = ["Manually generated demo clip (free web tier) - not an autonomous API generation"]

        source_path = self._find_clip(request.section_index, request.slot_index)
        if source_path is None:
            source_path = self._find_clip_for_section(request.section_index)
            if source_path is not None:
                warnings.append(
                    f"No exact demo clip for section {request.section_index} slot {request.slot_index} - "
                    "reused another clip from the same section (limited free-credit clip set)"
                )
        if source_path is None:
            source_path = self._find_any_clip()
            if source_path is not None:
                warnings.append(
                    f"No demo clip exists for section {request.section_index} at all - "
                    "reused a clip generated for a different section (limited free-credit clip set)"
                )

        if source_path is None:
            return AIVideoGenerationResult(
                success=False,
                status="failed",
                provider=self._provider_name,
                error=f"No local demo clips found at all in '{self.clips_dir}'",
            )

        return AIVideoGenerationResult(
            success=True,
            status="succeeded",
            provider=self._provider_name,
            model=self.DEMO_MODEL_LABEL,
            local_file_path=source_path,
            warnings=warnings,
        )

    def _list_clip_files(self) -> List[str]:
        if not os.path.isdir(self.clips_dir):
            return []
        return sorted(f for f in os.listdir(self.clips_dir) if os.path.isfile(os.path.join(self.clips_dir, f)))

    def _find_clip(self, section_index: int, slot_index: int) -> Optional[str]:
        prefix = f"{section_index}-{slot_index}."
        for filename in self._list_clip_files():
            if filename.startswith(prefix):
                return os.path.join(self.clips_dir, filename)
        return None

    def _find_clip_for_section(self, section_index: int) -> Optional[str]:
        prefix = f"{section_index}-"
        for filename in self._list_clip_files():
            if filename.startswith(prefix):
                return os.path.join(self.clips_dir, filename)
        return None

    def _find_any_clip(self) -> Optional[str]:
        files = self._list_clip_files()
        return os.path.join(self.clips_dir, files[0]) if files else None

    @staticmethod
    def copy_into_output(source_path: str, output_path: str) -> None:
        """Copy a served demo clip to the pipeline's own output location -
        a plain local file copy, never a network operation."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copyfile(source_path, output_path)
