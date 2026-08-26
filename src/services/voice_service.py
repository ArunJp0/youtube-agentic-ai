# Voice Generation Service: converts a ScriptResult into narration audio.
#
# This is a deterministic service, not an LLM-driven reasoning agent:
# narration text extraction and ordering is fixed logic with no model calls.
# Only the actual text-to-speech synthesis is delegated to a swappable
# VoiceProvider.
from __future__ import annotations

import difflib
import os
import uuid
from typing import List, Optional

from src.models.script import ScriptResult
from src.models.voice import VoiceResult
from src.tools.voice_provider import VoiceProvider

DEFAULT_OUTPUT_DIR = os.path.join("output", "audio")

# Similarity ratio (difflib.SequenceMatcher, on normalized text) at/above
# which two narration segments are treated as near-duplicates. Same
# calibration as ScriptAgent.SECTION_SIMILARITY_THRESHOLD: distinct content
# scores ~0.02-0.4, degenerate repeated content (shared boilerplate with a
# different lead-in) scores ~0.93.
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.82

# Minimum normalized length (characters) before the similarity ratio is
# trusted for fuzzy matching - see ScriptAgent.MIN_LENGTH_FOR_FUZZY_MATCH for
# why short strings need this guard. Exact duplicates are always caught
# regardless of length.
MIN_LENGTH_FOR_FUZZY_MATCH = 60


class VoiceServiceError(Exception):
    """Raised for configuration/programmer errors (e.g. missing input).

    Synthesis failures from the underlying VoiceProvider are NOT raised -
    they are captured in the returned VoiceResult (success=False, error=...)
    so callers always get a structured result back.
    """


class VoiceService:
    """Deterministic service that turns a ScriptResult into narration audio."""

    def __init__(
        self,
        voice_provider: VoiceProvider,
        voice_name: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        """Initialize the Voice Service.

        Args:
            voice_provider: Implementation of VoiceProvider for synthesis
            voice_name: Provider-specific voice identifier to use
            output_dir: Local directory to write generated audio files into
                (expected to be excluded from version control)
        """
        self.voice_provider = voice_provider
        self.voice_name = voice_name
        self.output_dir = output_dir

    @staticmethod
    def extract_narration(script: ScriptResult) -> str:
        """Extract narration text in playback order, then validate it.

        Order: hook -> introduction -> sections (in order) -> conclusion ->
        call_to_action - each exactly once. Source URLs, research notes,
        section headings, and visual notes are metadata for other consumers
        and are never narrated.

        As a final validation step before the text is handed to a
        VoiceProvider, any segment that is an exact or near-duplicate of an
        already-kept segment is dropped (first occurrence wins). This is a
        safety net independent of ScriptAgent's own section-level dedup -
        it protects against any ScriptResult (regardless of how it was
        produced) containing repeated content.

        Args:
            script: The ScriptResult to extract narration from

        Returns:
            Narration text ready to be sent to a VoiceProvider
        """
        segments: List[str] = [script.hook, script.introduction]
        segments.extend(section.narration for section in script.sections)
        segments.append(script.conclusion)
        segments.append(script.call_to_action)

        unique_segments = VoiceService._deduplicate_segments(segments)
        return "\n\n".join(unique_segments)

    @staticmethod
    def _deduplicate_segments(segments: List[str]) -> List[str]:
        """Drop exact/near-duplicate segments, keeping the first occurrence."""
        kept: List[str] = []
        for raw in segments:
            text = (raw or "").strip()
            if not text:
                continue
            if any(VoiceService._is_near_duplicate(text, kept_text) for kept_text in kept):
                continue
            kept.append(text)
        return kept

    @staticmethod
    def _is_near_duplicate(
        a: str, b: str, threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD
    ) -> bool:
        """True if two narration segments are exact or near-duplicates.

        Exact duplicates are always caught, regardless of length. Fuzzy
        matching only kicks in for paragraph-length text - see
        MIN_LENGTH_FOR_FUZZY_MATCH.
        """
        norm_a = " ".join(a.split()).strip().lower()
        norm_b = " ".join(b.split()).strip().lower()
        if not norm_a or not norm_b:
            return False
        if norm_a == norm_b:
            return True
        if len(norm_a) < MIN_LENGTH_FOR_FUZZY_MATCH or len(norm_b) < MIN_LENGTH_FOR_FUZZY_MATCH:
            return False
        return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold

    async def generate_voice(self, script: ScriptResult) -> VoiceResult:
        """Generate narration audio for a ScriptResult.

        Never raises for synthesis failures - those are captured in the
        returned VoiceResult. Only raises VoiceServiceError for
        configuration/programmer errors (e.g. a missing ScriptResult).

        Args:
            script: Structured script produced by the Script Agent

        Returns:
            Structured VoiceResult describing the outcome
        """
        if script is None:
            raise VoiceServiceError("ScriptResult is required")

        audio_format = self.voice_provider.output_format
        provider_name = self.voice_provider.name
        narration_text = self.extract_narration(script)

        if not narration_text:
            return VoiceResult(
                audio_file_path=None,
                provider=provider_name,
                voice_name=self.voice_name,
                duration_seconds=None,
                format=audio_format,
                success=False,
                error="No narration text to synthesize",
            )

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self._build_filename(script, audio_format))

        try:
            duration_seconds: Optional[float] = await self.voice_provider.synthesize(
                text=narration_text,
                voice_name=self.voice_name,
                output_path=output_path,
            )
        except Exception as e:
            return VoiceResult(
                audio_file_path=None,
                provider=provider_name,
                voice_name=self.voice_name,
                duration_seconds=None,
                format=audio_format,
                success=False,
                error=f"Voice synthesis failed: {e}",
            )

        return VoiceResult(
            audio_file_path=output_path,
            provider=provider_name,
            voice_name=self.voice_name,
            duration_seconds=duration_seconds,
            format=audio_format,
            success=True,
            error=None,
        )

    @staticmethod
    def _build_filename(script: ScriptResult, audio_format: str) -> str:
        slug = "".join(c if c.isalnum() else "-" for c in script.video_title.lower())
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug.strip("-")[:40] or "narration"
        return f"{slug}-{uuid.uuid4().hex[:8]}.{audio_format}"
