# Reconstructs a minimal ScriptResult from an existing .srt transcript when
# no real ScriptResult is available - no ScriptResult is ever persisted to
# disk anywhere in this project. Shared by every standalone demo that needs
# real script/narration context without re-running Research/Script
# (src/bgm_demo.py, src/metadata_demo.py) - one implementation, not a copy
# per demo.
#
# The reconstructed narration text is real (the actual spoken words, via
# CaptionService's own transcription), just not organized into the
# original script's real section boundaries - it's a best-effort stand-in
# for standalone validation only, never used by the main pipeline (which
# always has the real ScriptResult already in PipelineState).
from __future__ import annotations

import os
import re
from typing import Optional

from src.models.script import ScriptResult, ScriptSection

# VideoAssemblyService names output files "<slug>-<8-char-hex>.mp4" - this
# suffix (plus any "-captioned"/"-bgm" chain appended by later stages) is
# stripped to recover the original slug.
_ID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")

_KNOWN_SUFFIXES = ("-captioned-bgm", "-bgm", "-captioned")

RECONSTRUCTED_PLACEHOLDER = "(reconstructed for standalone validation - not part of the real script)"


def original_base_name(video_path: str) -> str:
    """The base filename CaptionService/AudioMixingService derived their
    .srt/-captioned/-bgm names from, regardless of which stage's output
    ``video_path`` actually points to."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    for suffix in _KNOWN_SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def find_matching_srt(video_path: str, subtitle_dir: str) -> Optional[str]:
    """Locate the .srt transcript CaptionService produced for this same
    video, if any (see CaptionService._build_srt_filename)."""
    srt_path = os.path.join(subtitle_dir, f"{original_base_name(video_path)}.srt")
    return srt_path if os.path.exists(srt_path) else None


def extract_narration_from_srt(srt_path: str) -> str:
    """Concatenate every caption's text into one narration string, in
    chronological order - ignores block indices/timestamps entirely."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    narration_parts = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        if text_lines:
            narration_parts.append(" ".join(text_lines))
    return " ".join(narration_parts).strip()


def derive_title_from_base_name(base_name: str) -> str:
    slug = _ID_SUFFIX_RE.sub("", base_name)
    words = [w for w in slug.replace("_", "-").split("-") if w]
    return " ".join(w.capitalize() for w in words) if words else base_name


def build_context_from_srt(topic: str, srt_path: str, video_path: str) -> ScriptResult:
    """Reconstruct a minimal but real-narration-grounded ScriptResult from
    an existing .srt transcript - never re-runs Research/Script."""
    narration = extract_narration_from_srt(srt_path) or topic
    title = derive_title_from_base_name(original_base_name(video_path))
    hook = narration[:200].rsplit(" ", 1)[0] if len(narration) > 200 else narration
    return ScriptResult(
        topic=topic,
        video_title=title,
        hook=hook or topic,
        introduction=RECONSTRUCTED_PLACEHOLDER,
        sections=[ScriptSection(heading="Full narration (from existing captions)", narration=narration)],
        conclusion=RECONSTRUCTED_PLACEHOLDER,
        call_to_action=RECONSTRUCTED_PLACEHOLDER,
    )


def build_topic_only_context(topic: str) -> ScriptResult:
    """Fallback when no matching .srt exists either - a weaker mood/content
    signal, but still real (the topic itself), never fabricated detail."""
    return ScriptResult(
        topic=topic,
        video_title=topic,
        hook=topic,
        introduction=RECONSTRUCTED_PLACEHOLDER,
        sections=[ScriptSection(heading="Topic only (no existing narration context found)", narration=topic)],
        conclusion=RECONSTRUCTED_PLACEHOLDER,
        call_to_action=RECONSTRUCTED_PLACEHOLDER,
    )
