# Real local speech transcription via faster-whisper.
#
# faster-whisper was chosen over the reference openai-whisper package: it
# is CTranslate2-backed, has no PyTorch dependency (a multi-GB install for
# what is otherwise a small local inference task), supports CPU inference
# with int8 quantization out of the box, and is actively maintained. Model
# weights are downloaded once (from Hugging Face) and cached locally on
# first use - no paid API, no per-request cost.
from __future__ import annotations

import os
from typing import List

from src.tools.transcription_provider import (
    TranscribedSegment,
    TranscribedWord,
    TranscriptionProvider,
    TranscriptionProviderError,
)

# "base" balances accuracy and speed/size for clean, single-speaker TTS
# narration (not noisy real-world audio) - larger models are unnecessary
# for this MVP's use case. Configurable via WHISPER_MODEL_SIZE.
DEFAULT_WHISPER_MODEL_SIZE = "base"
DEFAULT_WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"


class WhisperTranscriptionProvider(TranscriptionProvider):
    """Transcription provider backed by a local faster-whisper model.

    All faster-whisper-specific usage (model loading, word-timestamp
    extraction) is contained here; CaptionService and the rest of the app
    only ever see the plain TranscriptionProvider interface. The model is
    loaded lazily on first use (not at construction time) and cached for
    reuse across calls on the same provider instance.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_WHISPER_MODEL_SIZE,
        device: str = DEFAULT_WHISPER_DEVICE,
        compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def name(self) -> str:
        return "whisper"

    def transcribe(self, audio_path: str) -> List[TranscribedSegment]:
        """Transcribe ``audio_path`` with word-level timestamps where available.

        Raises:
            TranscriptionProviderError: If faster-whisper is not installed,
                the audio file is missing, or transcription fails.
        """
        if not audio_path or not os.path.exists(audio_path):
            raise TranscriptionProviderError(f"Audio file not found: {audio_path}")

        model = self._get_model()

        try:
            segments_iter, _info = model.transcribe(audio_path, word_timestamps=True)
            segments: List[TranscribedSegment] = []
            for segment in segments_iter:
                words = [
                    TranscribedWord(
                        word=word.word.strip(), start_seconds=word.start, end_seconds=word.end
                    )
                    for word in (segment.words or [])
                    if word.word and word.word.strip()
                ]
                segments.append(
                    TranscribedSegment(
                        text=segment.text.strip(),
                        start_seconds=segment.start,
                        end_seconds=segment.end,
                        words=words,
                    )
                )
            return segments
        except TranscriptionProviderError:
            raise
        except Exception as e:
            raise TranscriptionProviderError(f"Whisper transcription failed: {e}") from e

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscriptionProviderError(
                "faster-whisper is not installed. Install it with `pip install faster-whisper`."
            ) from e

        try:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception as e:
            raise TranscriptionProviderError(f"Failed to load Whisper model '{self.model_size}': {e}") from e
        return self._model
