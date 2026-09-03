# Tests for WhisperTranscriptionProvider. faster_whisper.WhisperModel is
# monkeypatched in every test in this module, so no real model download or
# inference ever happens.
from __future__ import annotations

import sys

import pytest

from src.tools.transcription_provider import TranscriptionProviderError
from src.tools.whisper_transcription_provider import (
    DEFAULT_WHISPER_MODEL_SIZE,
    WhisperTranscriptionProvider,
)


class FakeWord:
    def __init__(self, word: str, start: float, end: float) -> None:
        self.word = word
        self.start = start
        self.end = end


class FakeSegment:
    def __init__(self, text: str, start: float, end: float, words=None) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class FakeWhisperModel:
    instances_created = 0

    def __init__(self, model_size, device, compute_type) -> None:
        FakeWhisperModel.instances_created += 1
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[dict] = []

    def transcribe(self, audio_path, word_timestamps=True):
        self.transcribe_calls.append({"audio_path": audio_path, "word_timestamps": word_timestamps})
        segments = [
            FakeSegment(
                "Hello world.",
                0.0,
                1.0,
                words=[FakeWord("Hello", 0.0, 0.5), FakeWord(" ", 0.5, 0.5), FakeWord("world.", 0.5, 1.0)],
            )
        ]
        return iter(segments), object()


@pytest.fixture(autouse=True)
def _reset_fake_model_counter():
    FakeWhisperModel.instances_created = 0
    yield


def _patch_whisper_model(monkeypatch, model_cls=FakeWhisperModel) -> None:
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", model_cls)


class TestWhisperTranscriptionProviderInit:
    def test_name(self) -> None:
        assert WhisperTranscriptionProvider().name == "whisper"

    def test_default_model_size(self) -> None:
        assert WhisperTranscriptionProvider().model_size == DEFAULT_WHISPER_MODEL_SIZE

    def test_custom_model_size(self) -> None:
        assert WhisperTranscriptionProvider(model_size="tiny").model_size == "tiny"

    def test_model_not_loaded_at_construction(self) -> None:
        provider = WhisperTranscriptionProvider()
        assert provider._model is None


class TestWhisperTranscriptionProviderTranscribe:
    def test_missing_audio_file_raises(self, tmp_path) -> None:
        provider = WhisperTranscriptionProvider()
        with pytest.raises(TranscriptionProviderError, match="not found"):
            provider.transcribe(str(tmp_path / "missing.mp3"))

    def test_returns_segments_with_words(self, monkeypatch, tmp_path) -> None:
        _patch_whisper_model(monkeypatch)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider()
        segments = provider.transcribe(str(audio_path))

        assert len(segments) == 1
        assert segments[0].text == "Hello world."
        assert segments[0].start_seconds == 0.0
        assert segments[0].end_seconds == 1.0
        assert [w.word for w in segments[0].words] == ["Hello", "world."]

    def test_blank_words_filtered_out(self, monkeypatch, tmp_path) -> None:
        _patch_whisper_model(monkeypatch)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider()
        segments = provider.transcribe(str(audio_path))

        assert " " not in [w.word for w in segments[0].words]

    def test_word_timestamps_requested(self, monkeypatch, tmp_path) -> None:
        _patch_whisper_model(monkeypatch)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider()
        provider.transcribe(str(audio_path))

        model = provider._model
        assert model.transcribe_calls[0]["word_timestamps"] is True

    def test_model_loaded_lazily_and_cached_across_calls(self, monkeypatch, tmp_path) -> None:
        _patch_whisper_model(monkeypatch)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider()
        provider.transcribe(str(audio_path))
        provider.transcribe(str(audio_path))

        assert FakeWhisperModel.instances_created == 1

    def test_model_construction_uses_configured_params(self, monkeypatch, tmp_path) -> None:
        _patch_whisper_model(monkeypatch)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider(model_size="small", device="cpu", compute_type="int8")
        provider.transcribe(str(audio_path))

        assert provider._model.model_size == "small"
        assert provider._model.device == "cpu"
        assert provider._model.compute_type == "int8"

    def test_faster_whisper_not_installed_raises_clean_error(self, monkeypatch, tmp_path) -> None:
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        provider = WhisperTranscriptionProvider()
        with pytest.raises(TranscriptionProviderError, match="not installed"):
            provider.transcribe(str(audio_path))

    def test_model_load_failure_raises_clean_error(self, monkeypatch, tmp_path) -> None:
        class ExplodingModel:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("could not download model")

        _patch_whisper_model(monkeypatch, model_cls=ExplodingModel)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider()
        with pytest.raises(TranscriptionProviderError, match="Failed to load"):
            provider.transcribe(str(audio_path))

    def test_transcription_call_failure_raises_clean_error(self, monkeypatch, tmp_path) -> None:
        class ExplodingTranscribeModel(FakeWhisperModel):
            def transcribe(self, audio_path, word_timestamps=True):
                raise RuntimeError("decode error")

        _patch_whisper_model(monkeypatch, model_cls=ExplodingTranscribeModel)
        audio_path = tmp_path / "narration.mp3"
        audio_path.write_bytes(b"FAKE AUDIO")

        provider = WhisperTranscriptionProvider()
        with pytest.raises(TranscriptionProviderError, match="Whisper transcription failed"):
            provider.transcribe(str(audio_path))
