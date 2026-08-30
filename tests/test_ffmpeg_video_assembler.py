# Tests for FFmpegVideoAssembler. subprocess.run and shutil.which are
# mocked in every test in this module, so no real FFmpeg process is ever
# started and these tests pass whether or not FFmpeg is installed.
from __future__ import annotations

import subprocess

import pytest

from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError


class FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _which_found(name: str) -> str:
    return f"/usr/bin/{name}"


class TestFFmpegVideoAssemblerInit:
    def test_missing_ffmpeg_binary_raises_with_windows_instructions(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(VideoAssemblerError, match="winget"):
            FFmpegVideoAssembler()

    def test_missing_binary_error_names_ffmpeg_install(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(VideoAssemblerError, match="ffmpeg -version"):
            FFmpegVideoAssembler()

    def test_found_binaries_are_used(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        assert assembler.ffmpeg_path == "/usr/bin/ffmpeg"
        assert assembler.ffprobe_path == "/usr/bin/ffprobe"

    def test_explicit_paths_bypass_which(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        assembler = FFmpegVideoAssembler(ffmpeg_path="C:/ffmpeg.exe", ffprobe_path="C:/ffprobe.exe")
        assert assembler.ffmpeg_path == "C:/ffmpeg.exe"


class TestFFmpegVideoAssemblerProbe:
    def test_probe_duration_parses_ffprobe_output(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompletedProcess(stdout="12.345\n"))
        assert assembler.probe_duration_seconds("audio.mp3") == 12.345

    def test_probe_duration_invalid_output_raises(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompletedProcess(stdout="not-a-number"))
        with pytest.raises(VideoAssemblerError):
            assembler.probe_duration_seconds("audio.mp3")

    def test_probe_uses_correct_ffprobe_output_format_option(self, monkeypatch) -> None:
        """Regression test: 'noprint_wrappers' (not 'noprint_wrapped_key')
        is the real ffprobe option name - a typo here passed all other
        mocked tests but failed against real ffprobe 8.0."""
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeCompletedProcess(stdout="1.0")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assembler.probe_duration_seconds("audio.mp3")

        of_value = captured["cmd"][captured["cmd"].index("-of") + 1]
        assert of_value == "default=noprint_wrappers=1:nokey=1"

    def test_nonzero_exit_raises_with_stderr(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: FakeCompletedProcess(returncode=1, stderr="boom")
        )
        with pytest.raises(VideoAssemblerError, match="boom"):
            assembler.probe_duration_seconds("audio.mp3")


class TestFFmpegVideoAssemblerCommandConstruction:
    def test_build_section_clip_video_uses_stream_loop(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", fake_run)

        assembler.build_section_clip("in.mp4", "out.mp4", 5.0, 1920, 1080, 30, is_video=True)

        cmd = captured["cmd"]
        assert "-stream_loop" in cmd
        assert "-1" in cmd
        assert "-t" in cmd
        assert "5.000" in cmd
        assert any("scale=1920:1080" in part for part in cmd)
        assert any("crop=1920:1080" in part for part in cmd)
        assert "-an" in cmd
        assert "libx264" in cmd
        assert "out.mp4" == cmd[-1]

    def test_build_section_clip_image_uses_loop_not_stream_loop(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", fake_run)

        assembler.build_section_clip("in.jpg", "out.mp4", 3.0, 1920, 1080, 30, is_video=False)

        cmd = captured["cmd"]
        assert "-loop" in cmd
        assert "-stream_loop" not in cmd

    def test_build_section_clip_preserves_aspect_ratio_via_increase_and_crop(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", fake_run)

        assembler.build_section_clip("in.jpg", "out.mp4", 4.0, 1280, 720, 24, is_video=False)

        vf_value = captured["cmd"][captured["cmd"].index("-vf") + 1]
        assert "force_original_aspect_ratio=increase" in vf_value
        assert "crop=1280:720" in vf_value
        assert "24" in captured["cmd"]  # fps

    def test_concatenate_and_mux_audio_command_shape(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", fake_run)

        assembler.concatenate_and_mux_audio(
            ["clip1.mp4", "clip2.mp4"], "narration.mp3", str(tmp_path / "out.mp4"), str(tmp_path)
        )

        assert len(commands) == 2
        concat_cmd, mux_cmd = commands
        assert "-f" in concat_cmd and "concat" in concat_cmd
        assert "-safe" in concat_cmd and "0" in concat_cmd
        assert "-shortest" in mux_cmd
        assert "aac" in mux_cmd
        assert "narration.mp3" in mux_cmd

        concat_list = (tmp_path / "concat_list.txt").read_text()
        assert concat_list.index("clip1.mp4") < concat_list.index("clip2.mp4")


class TestFFmpegVideoAssemblerErrorHandling:
    def test_ffmpeg_binary_disappears_between_check_and_run(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("gone")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(VideoAssemblerError):
            assembler.probe_duration_seconds("x.mp3")

    def test_timeout_raises_clean_error(self, monkeypatch) -> None:
        monkeypatch.setattr("shutil.which", _which_found)
        assembler = FFmpegVideoAssembler()

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(VideoAssemblerError, match="timed out"):
            assembler.probe_duration_seconds("x.mp3")
