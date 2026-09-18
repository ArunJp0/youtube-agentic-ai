# Tests for NeutralVisualGenerator/FFmpegNeutralVisualGenerator: the
# locally-generated, non-claim-specific last-resort Visual QC fallback (see
# src.services.visual_qc_service). Real FFmpeg tests are gated on the
# binary actually being available, mirroring test_ffmpeg_video_assembler.py's
# own convention - no real FFmpeg call in the unconditional/unit-level tests.
from __future__ import annotations

import shutil

import pytest

from src.tools.neutral_visual_generator import (
    FFmpegNeutralVisualGenerator,
    NeutralVisualGenerator,
    NeutralVisualGeneratorError,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def test_generator_is_an_abstract_interface() -> None:
    with pytest.raises(TypeError):
        NeutralVisualGenerator()  # type: ignore[abstract]


def test_missing_ffmpeg_binary_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(NeutralVisualGeneratorError):
        FFmpegNeutralVisualGenerator()


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="Real ffmpeg binary not available")
def test_real_ffmpeg_generates_a_playable_neutral_clip(tmp_path) -> None:
    generator = FFmpegNeutralVisualGenerator()
    output_path = str(tmp_path / "neutral.mp4")

    generator.generate(output_path=output_path, duration_seconds=1.0, width=320, height=180, fps=10, label="Test Section")

    import os

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="Real ffmpeg binary not available")
def test_real_ffmpeg_generates_without_a_label(tmp_path) -> None:
    generator = FFmpegNeutralVisualGenerator()
    output_path = str(tmp_path / "neutral_no_label.mp4")

    generator.generate(output_path=output_path, duration_seconds=1.0, width=320, height=180, fps=10, label=None)

    import os

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="Real ffmpeg binary not available")
def test_real_ffmpeg_escapes_special_characters_in_label(tmp_path) -> None:
    """A label containing filtergraph-meaningful characters (colon, single
    quote) must not break the FFmpeg command."""
    generator = FFmpegNeutralVisualGenerator()
    output_path = str(tmp_path / "neutral_special.mp4")

    generator.generate(
        output_path=output_path, duration_seconds=1.0, width=320, height=180, fps=10,
        label="Section: What's Next?",
    )

    import os

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
