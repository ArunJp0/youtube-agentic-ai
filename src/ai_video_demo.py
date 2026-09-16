# Standalone, demo-specific client showcase: assembles ONE short
# (~45-60s) video using the single real PixVerse AI-generated clip already
# produced manually (output/ai_video_demo/why-do-cats-purr/0-0.mp4), to
# prove genuine AI-generated footage can flow through this project's real
# production components end to end.
#
# NOT part of the normal production pipeline (src/workflows/pipeline_graph.py
# is entirely untouched) and NOT the AI-video pipeline INTEGRATION added
# earlier (VisualMediaService's ai_video_provider seam) - this script exists
# only because we have exactly ONE 5-second source clip and PixVerse free
# credits are exhausted, which the normal duration-aware multi-slot visual
# planning was never designed to stretch across a whole video by itself.
#
# Reuses, completely unmodified: VoiceService (real Edge TTS), VideoAssemblyService
# + FFmpegVideoAssembler (real per-section trim/loop/scale/crop/concat/mux -
# including its own real image-vs-video handling, used here to turn one
# extracted still frame into a held "freeze frame"/title-card segment),
# CaptionService (real local Whisper transcription + burn), AudioMixingService
# (real local BGM catalog, deterministic mood fallback - no LLM call). The
# only new code is src/services/ai_demo_visual_variety.py (a handful of
# deterministic FFmpeg crop/speed transforms of the ONE source clip, so the
# assembled video doesn't look like the same footage looping) and this
# orchestration script itself.
#
# Hand-composed narration (not a fresh ScriptAgent/Gemini call) grounded in
# the REAL research already gathered for "Why do cats purr?" in the prior
# milestone - no new external research/LLM call needed for this demo.
from __future__ import annotations

import asyncio
import os
import sys

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from src.config.settings import Settings
from src.models.media import MediaAsset, SectionMediaMapping, VisualResult
from src.models.script import ScriptResult, ScriptSection
from src.models.video import VideoAssemblyResult
from src.services.ai_demo_visual_variety import (
    CENTER_SLOW_ZOOM,
    LEFT_REFRAME,
    VisualVarietyError,
    generate_variant_clip,
    resolve_ffmpeg_path,
)
from src.services.audio_mixing_service import AudioMixingService
from src.services.caption_service import CaptionService
from src.services.video_assembly_service import VideoAssemblyService
from src.services.voice_service import VoiceService
from src.tools.ffmpeg_video_assembler import FFmpegVideoAssembler, VideoAssemblerError
from src.tools.music_catalog_provider import LocalMusicCatalogProvider
from src.tools.edge_voice_provider import EdgeVoiceProvider
from src.tools.whisper_transcription_provider import WhisperTranscriptionProvider

DEMO_DIR = os.path.join("output", "ai_video_demo", "why-do-cats-purr")
SOURCE_CLIP_PATH = os.path.join(DEMO_DIR, "0-0.mp4")
FINAL_OUTPUT_PATH = os.path.join(DEMO_DIR, "final_ai_demo.mp4")
WORK_DIR = os.path.join(DEMO_DIR, "work")

TOPIC = "Why do cats purr?"

# Hand-composed, grounded in the real research already produced for this
# topic in the prior milestone (nasal-vaccine-unrelated - this is the cats
# research: purring's individual acoustic signature, kneading/comfort
# behavior, social greeting function, and the stress/frustration response -
# all real facts already gathered, not invented for this demo). hook/
# introduction/call_to_action are intentionally a single space: ScriptResult
# requires non-empty strings, but VoiceService.extract_narration strips and
# drops blank segments, so nothing is actually spoken for these fields -
# every real spoken word lives in sections[*].narration/conclusion,
# deliberately, so each has a precise, controllable share of the video's
# visual timeline (see calculate_section_durations).
DEMO_SCRIPT = ScriptResult(
    topic=TOPIC,
    video_title="Why Do Cats Purr?",
    hook=" ",
    introduction=" ",
    sections=[
        ScriptSection(
            heading="Why Do Cats Purr?",
            narration="Ever wondered what that gentle rumble coming from your cat actually means?",
        ),
        ScriptSection(
            heading="Individual Purr Signature",
            narration=(
                "Purring is one of the most common sounds an adult cat makes, and remarkably, "
                "each cat's purr carries its own unique acoustic signature, almost like a vocal "
                "fingerprint no two cats fully share."
            ),
        ),
        ScriptSection(
            heading="Comfort and Kneading",
            narration=(
                "Cats often purr while kneading a soft blanket or curling up close to you, a "
                "soothing habit rooted in comfort, safety, and social bonding."
            ),
        ),
        ScriptSection(
            heading="More Than Happiness",
            narration=(
                "But purring isn't only about contentment. Cats also purr as a friendly greeting "
                "toward other cats, and surprisingly, even when they're stressed or anxious, "
                "making it a far more complex signal than simple happiness."
            ),
        ),
    ],
    conclusion=(
        "So the next time your cat purrs, it might be telling you something far more interesting "
        "than simple happiness, and worth listening to a little closer."
    ),
    call_to_action=" ",
    sources=[],
)


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _resolve_demo_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_centered_text(image: Image.Image, text: str, size: int, y_fraction: float) -> None:
    draw = ImageDraw.Draw(image)
    font = _resolve_demo_font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image.width - text_w) / 2
    y = image.height * y_fraction - text_h / 2
    # Simple readable stroke outline - no filters/gimmicks, matching the
    # existing project's thumbnail text-rendering style (contrast, not flash).
    draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))


def _build_title_card(frame_path: str, output_path: str) -> str:
    """A blurred/darkened still from the REAL source clip with clean
    centered title typography - ties the opening card visually to the real
    AI footage rather than an unrelated graphic."""
    image = Image.open(frame_path).convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=6))
    image = ImageEnhance.Brightness(image).enhance(0.55)
    _draw_centered_text(image, "Why Do Cats Purr?", size=88, y_fraction=0.5)
    image.save(output_path, format="JPEG", quality=92)
    return output_path


def _build_freeze_frame_card(frame_path: str, output_path: str) -> str:
    """A real still from the source clip with a small caption-style label -
    the 'typography/caption-led moment' + freeze-frame technique."""
    image = Image.open(frame_path).convert("RGB")
    _draw_centered_text(image, "Not just happiness...", size=54, y_fraction=0.85)
    image.save(output_path, format="JPEG", quality=92)
    return output_path


async def run_ai_video_demo() -> dict:
    _ensure_utf8_stdout()
    print(f"AI Video Demo: {TOPIC}")
    print("=" * 60)

    # ---- STEP 1: pre-flight validation --------------------------------
    if not os.path.exists(SOURCE_CLIP_PATH):
        raise FileNotFoundError(f"Source AI clip not found: {SOURCE_CLIP_PATH}")
    if os.path.getsize(SOURCE_CLIP_PATH) == 0:
        raise ValueError(f"Source AI clip is empty: {SOURCE_CLIP_PATH}")
    ffmpeg_path = resolve_ffmpeg_path()
    print(f"[OK] Source clip validated: {SOURCE_CLIP_PATH}")

    stock_fallback_enabled_for_this_demo = False  # hardcoded for THIS script only - .env untouched
    youtube_publishing_enabled_for_this_demo = False
    print(f"[OK] Stock/Pexels fallback disabled for this demo: {not stock_fallback_enabled_for_this_demo}")
    print(f"[OK] YouTube publishing disabled for this demo: {not youtube_publishing_enabled_for_this_demo}")
    print("=" * 60)

    os.makedirs(WORK_DIR, exist_ok=True)
    settings = Settings()
    assembler = FFmpegVideoAssembler()

    # ---- STEP 2: generate visual variety from the ONE real clip -------
    print("Generating deterministic visual variants from the single real AI clip...")
    center_variant_path = os.path.join(WORK_DIR, "variant-center-slow.mp4")
    left_variant_path = os.path.join(WORK_DIR, "variant-left-reframe.mp4")
    try:
        generate_variant_clip(SOURCE_CLIP_PATH, center_variant_path, CENTER_SLOW_ZOOM, ffmpeg_path)
        generate_variant_clip(SOURCE_CLIP_PATH, left_variant_path, LEFT_REFRAME, ffmpeg_path)
    except VisualVarietyError as e:
        raise RuntimeError(f"Failed to generate visual variants: {e}") from e

    frame_paths = assembler.extract_frames(
        SOURCE_CLIP_PATH, timestamps_seconds=[0.1, 2.5], output_dir=WORK_DIR, basename="frame"
    )
    title_card_path = _build_title_card(frame_paths[0], os.path.join(WORK_DIR, "title-card.jpg"))
    freeze_frame_path = _build_freeze_frame_card(frame_paths[1], os.path.join(WORK_DIR, "freeze-frame.jpg"))
    print("[OK] Visual variants ready: title card, slow center zoom, left reframe, freeze frame")
    print("=" * 60)

    # ---- STEP 3: real narration audio (Voice Service, unmodified) -----
    print("Synthesizing real narration audio (Edge TTS)...")
    voice_service = VoiceService(
        voice_provider=EdgeVoiceProvider(), voice_name=settings.voice_name, output_dir=WORK_DIR
    )
    voice_result = await voice_service.generate_voice(DEMO_SCRIPT)
    if not voice_result.success:
        raise RuntimeError(f"Voice synthesis failed: {voice_result.error}")
    print(f"[OK] Narration audio: {voice_result.audio_file_path} ({voice_result.duration_seconds:.1f}s)")
    print("=" * 60)

    # ---- STEP 4: assemble the video (Video Assembly Service, unmodified) ----
    print("Assembling video from AI-clip-only visual assets (Video Assembly Service)...")
    visual_result = VisualResult(
        topic=TOPIC,
        provider="local_ai_video",
        success=True,
        sections=[
            SectionMediaMapping(
                section_index=0,
                section_heading=DEMO_SCRIPT.sections[0].heading,
                assets=[
                    MediaAsset(
                        provider="local_ai_video",
                        asset_type="image",
                        local_file_path=title_card_path,
                        search_query="title card",
                        section_index=0,
                        success=True,
                        relevance_tier="ai_generated",
                    )
                ],
            ),
            SectionMediaMapping(
                section_index=1,
                section_heading=DEMO_SCRIPT.sections[1].heading,
                assets=[
                    MediaAsset(
                        provider="local_ai_video",
                        asset_type="video",
                        local_file_path=center_variant_path,
                        search_query="center slow zoom",
                        section_index=1,
                        success=True,
                        relevance_tier="ai_generated",
                    )
                ],
            ),
            SectionMediaMapping(
                section_index=2,
                section_heading=DEMO_SCRIPT.sections[2].heading,
                assets=[
                    MediaAsset(
                        provider="local_ai_video",
                        asset_type="video",
                        local_file_path=left_variant_path,
                        search_query="left reframe",
                        section_index=2,
                        success=True,
                        relevance_tier="ai_generated",
                    )
                ],
            ),
            SectionMediaMapping(
                section_index=3,
                section_heading=DEMO_SCRIPT.sections[3].heading,
                assets=[
                    MediaAsset(
                        provider="local_ai_video",
                        asset_type="image",
                        local_file_path=freeze_frame_path,
                        search_query="freeze frame",
                        section_index=3,
                        success=True,
                        relevance_tier="ai_generated",
                    )
                ],
            ),
        ],
    )
    video_service = VideoAssemblyService(assembler=assembler, output_dir=WORK_DIR)
    video_result = await video_service.assemble_video(DEMO_SCRIPT, voice_result, visual_result)
    if not video_result.success:
        raise RuntimeError(f"Video assembly failed: {video_result.error}")
    print(f"[OK] Assembled (silent narration muxed): {video_result.output_path} ({video_result.duration_seconds}s)")
    print("=" * 60)

    # ---- STEP 5: real captions (Caption Service, unmodified) -----------
    print("Burning real captions (local Whisper transcription)...")
    caption_service = CaptionService(
        transcription_provider=WhisperTranscriptionProvider(model_size=settings.whisper_model_size),
        assembler=assembler,
        output_dir=WORK_DIR,
    )
    caption_result = await caption_service.generate_captions(voice_result, video_result)
    if not caption_result.success:
        raise RuntimeError(f"Captioning failed: {caption_result.error}")
    print(f"[OK] Captioned: {caption_result.captioned_video_path}")
    print("=" * 60)

    # ---- STEP 6: real BGM (Audio Mixing Service, unmodified) -----------
    print("Mixing local approved BGM (deterministic mood fallback, no LLM call)...")
    audio_mix_service = AudioMixingService(
        catalog_provider=LocalMusicCatalogProvider(),
        assembler=assembler,
        llm_provider=None,  # no unnecessary Gemini call - deterministic fallback plan
    )
    # BGM is mixed onto the captioned output (what viewers will actually
    # see), not the pre-caption assembly - the exact same adapter shape
    # the production pipeline's own _captioned_video_result helper builds
    # (src/workflows/pipeline_graph.py), not a new pattern invented here.
    captioned_video_result = VideoAssemblyResult(
        success=True,
        output_path=caption_result.captioned_video_path,
        duration_seconds=caption_result.captioned_duration_seconds,
        format="mp4",
    )
    mix_result = await audio_mix_service.generate_mix(TOPIC, DEMO_SCRIPT, captioned_video_result)
    if not mix_result.success:
        raise RuntimeError(f"BGM mixing failed: {mix_result.error}")
    print(f"[OK] BGM mixed: {mix_result.output_path} (track: {mix_result.selected_track.title if mix_result.selected_track else None})")
    print("=" * 60)

    # ---- STEP 7: save as the final demo output -------------------------
    os.makedirs(DEMO_DIR, exist_ok=True)
    import shutil

    shutil.copyfile(mix_result.output_path, FINAL_OUTPUT_PATH)
    print(f"[OK] Final demo saved: {FINAL_OUTPUT_PATH}")

    return {
        "final_output_path": FINAL_OUTPUT_PATH,
        "voice_result": voice_result,
        "video_result": video_result,
        "caption_result": caption_result,
        "mix_result": mix_result,
    }


if __name__ == "__main__":
    try:
        asyncio.run(run_ai_video_demo())
    except (FileNotFoundError, ValueError, RuntimeError, VideoAssemblerError, VisualVarietyError) as e:
        print(f"AI video demo failed: {e}")
        sys.exit(1)
