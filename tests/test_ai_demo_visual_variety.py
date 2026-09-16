# Tests for src.services.ai_demo_visual_variety - pure FFmpeg command
# construction only. No real FFmpeg process is ever invoked in this file;
# run_ffmpeg_command/generate_variant_clip's actual subprocess execution is
# exercised only by the real, explicitly-approved demo run, never by pytest.
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.ai_demo_visual_variety import (
    CENTER_SLOW_ZOOM,
    LEFT_REFRAME,
    RIGHT_REFRAME,
    CropSpeedVariant,
    VisualVarietyError,
    build_variant_command,
    generate_variant_clip,
    resolve_ffmpeg_path,
)


class TestBuildVariantCommand:
    def test_includes_input_and_output_paths(self) -> None:
        command = build_variant_command("in.mp4", "out.mp4", CENTER_SLOW_ZOOM)
        assert "in.mp4" in command
        assert "out.mp4" in command

    def test_crop_filter_present_in_video_filter_argument(self) -> None:
        command = build_variant_command("in.mp4", "out.mp4", LEFT_REFRAME)
        vf_index = command.index("-vf") + 1
        assert LEFT_REFRAME.crop_expr in command[vf_index]

    def test_speed_factor_of_one_omits_setpts(self) -> None:
        no_speed_change = CropSpeedVariant(name="x", crop_expr="crop=in_w:in_h:0:0", speed_factor=1.0)
        command = build_variant_command("in.mp4", "out.mp4", no_speed_change)
        vf_index = command.index("-vf") + 1
        assert "setpts" not in command[vf_index]

    def test_slower_speed_factor_adds_setpts_scaling_up(self) -> None:
        command = build_variant_command("in.mp4", "out.mp4", CENTER_SLOW_ZOOM)
        vf_index = command.index("-vf") + 1
        assert "setpts=1.1765*PTS" in command[vf_index]  # 1 / 0.85

    def test_no_audio_stream_in_output(self) -> None:
        command = build_variant_command("in.mp4", "out.mp4", CENTER_SLOW_ZOOM)
        assert "-an" in command

    def test_deterministic_same_inputs_produce_identical_command(self) -> None:
        first = build_variant_command("in.mp4", "out.mp4", RIGHT_REFRAME)
        second = build_variant_command("in.mp4", "out.mp4", RIGHT_REFRAME)
        assert first == second

    def test_different_variants_produce_different_crop_filters(self) -> None:
        left = build_variant_command("in.mp4", "out.mp4", LEFT_REFRAME)
        right = build_variant_command("in.mp4", "out.mp4", RIGHT_REFRAME)
        assert left != right

    def test_custom_ffmpeg_path_used_as_executable(self) -> None:
        command = build_variant_command("in.mp4", "out.mp4", CENTER_SLOW_ZOOM, ffmpeg_path="/custom/ffmpeg")
        assert command[0] == "/custom/ffmpeg"


class TestGenerateVariantClip:
    def test_delegates_to_run_ffmpeg_command_with_built_command(self) -> None:
        with patch("src.services.ai_demo_visual_variety.run_ffmpeg_command") as mock_run:
            result = generate_variant_clip("in.mp4", "out.mp4", LEFT_REFRAME)

        assert result == "out.mp4"
        mock_run.assert_called_once()
        called_command = mock_run.call_args[0][0]
        assert called_command == build_variant_command("in.mp4", "out.mp4", LEFT_REFRAME)

    def test_failure_propagates_as_visual_variety_error(self) -> None:
        with patch(
            "src.services.ai_demo_visual_variety.run_ffmpeg_command", side_effect=VisualVarietyError("boom")
        ):
            with pytest.raises(VisualVarietyError):
                generate_variant_clip("in.mp4", "out.mp4", LEFT_REFRAME)


class TestResolveFfmpegPath:
    def test_missing_executable_raises_clear_error(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(VisualVarietyError, match="not found on PATH"):
                resolve_ffmpeg_path("definitely-not-a-real-binary")

    def test_found_executable_returns_resolved_path(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert resolve_ffmpeg_path("ffmpeg") == "/usr/bin/ffmpeg"
