# Deterministic thumbnail rendering (Pillow). No LLM/network involved here
# - the ThumbnailPlan's `composition` field selects one of a small, fixed
# set of layout rules; everything else (crop math, text wrapping, font
# size, contrast panel) is computed deterministically. The LLM never
# supplies pixel coordinates.
#
# Note: this project does no ML-based subject detection - `composition`
# only controls which side of the frame hosts the text panel (and the
# translucent scrim that guarantees contrast there), not literal awareness
# of where a photo's subject actually is. That's an accepted MVP
# simplification, not a bug.
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.models.thumbnail import ThumbnailPlan

THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720

SAFE_MARGIN_PX = 60
MAX_HOOK_FONT_SIZE = 96
MIN_HOOK_FONT_SIZE = 40
FONT_SIZE_STEP = 4

TEXT_COLOR = (255, 255, 255)
STROKE_COLOR = (0, 0, 0)
STROKE_WIDTH = 6
OVERLAY_COLOR = (0, 0, 0)
OVERLAY_OPACITY = 130  # 0-255 alpha for the contrast panel behind text

# Configurable via ThumbnailAgent's font_path constructor arg; these are
# safe, commonly-available local fallbacks tried in order after it - never
# downloaded, never require a design application to be open.
_FONT_FALLBACK_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


class ThumbnailRenderError(Exception):
    """Raised when deterministic rendering fails (bad source image, or the
    output file can't be written)."""


def _resolve_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    for path in ([font_path] if font_path else []) + _FONT_FALLBACK_CANDIDATES:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    # Last resort: Pillow's bundled bitmap font - always available, so
    # rendering can never fail purely because no TTF was found.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _crop_to_aspect(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop (never stretch) to the target aspect ratio, then resize
    to exactly (target_w, target_h). Handles both landscape and portrait
    sources safely."""
    src_w, src_h = image.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = max(int(src_h * target_ratio), 1)
        left = (src_w - new_w) // 2
        box = (left, 0, left + new_w, src_h)
    else:
        new_h = max(int(src_w / target_ratio), 1)
        top = (src_h - new_h) // 2
        box = (0, top, src_w, top + new_h)

    return image.crop(box).resize((target_w, target_h), Image.LANCZOS)


def _layout_for_composition(composition: str) -> dict:
    """A small, fixed set of deterministic layouts - the only thing
    ``composition`` (an LLM-supplied but closed-vocabulary field) controls."""
    if composition == "subject_left":
        return {
            "panel": (THUMBNAIL_WIDTH // 2, 0, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT),
            "text_box": (
                THUMBNAIL_WIDTH // 2 + SAFE_MARGIN_PX // 2,
                SAFE_MARGIN_PX,
                THUMBNAIL_WIDTH - SAFE_MARGIN_PX,
                THUMBNAIL_HEIGHT - SAFE_MARGIN_PX,
            ),
            "align": "left",
        }
    if composition == "subject_right":
        return {
            "panel": (0, 0, THUMBNAIL_WIDTH // 2, THUMBNAIL_HEIGHT),
            "text_box": (
                SAFE_MARGIN_PX,
                SAFE_MARGIN_PX,
                THUMBNAIL_WIDTH // 2 - SAFE_MARGIN_PX // 2,
                THUMBNAIL_HEIGHT - SAFE_MARGIN_PX,
            ),
            "align": "left",
        }
    # centered: a bottom band spanning most of the width, like a lower third.
    band_top = int(THUMBNAIL_HEIGHT * 0.68)
    return {
        "panel": (0, band_top, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT),
        "text_box": (
            SAFE_MARGIN_PX,
            band_top + SAFE_MARGIN_PX // 2,
            THUMBNAIL_WIDTH - SAFE_MARGIN_PX,
            THUMBNAIL_HEIGHT - SAFE_MARGIN_PX // 2,
        ),
        "align": "center",
    }


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [text] if text else []
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_text(
    draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int, font_path: Optional[str]
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    """Deterministically reduce font size until the wrapped text fits
    within max_width/max_height - text NEVER overflows the canvas."""
    size = MAX_HOOK_FONT_SIZE
    font = _resolve_font(font_path, size)
    lines = _wrap_text(draw, text, font, max_width)

    while size > MIN_HOOK_FONT_SIZE:
        font = _resolve_font(font_path, size)
        lines = _wrap_text(draw, text, font, max_width)
        line_height = _line_height(draw, font)
        total_height = line_height * len(lines)
        widest = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0)
        if total_height <= max_height and widest <= max_width:
            break
        size -= FONT_SIZE_STEP
    else:
        font = _resolve_font(font_path, MIN_HOOK_FONT_SIZE)
        lines = _wrap_text(draw, text, font, max_width)

    # Absolute last-resort safety net: even at the minimum size, clip to
    # however many lines actually fit rather than let text visually
    # overflow the canvas.
    line_height = _line_height(draw, font)
    max_lines = max(int(max_height // line_height), 1)
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    return font, lines


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> int:
    return draw.textbbox((0, 0), "Ag", font=font)[3] + 14


def _draw_lines(draw: ImageDraw.ImageDraw, lines: List[str], font: ImageFont.FreeTypeFont, layout: dict) -> None:
    x0, y0, x1, y1 = layout["text_box"]
    line_height = _line_height(draw, font)
    total_height = line_height * len(lines)
    start_y = y0 + max((y1 - y0 - total_height) // 2, 0)

    for index, line in enumerate(lines):
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        if layout["align"] == "center":
            x = x0 + max((x1 - x0 - line_width) // 2, 0)
        else:
            x = x0
        y = start_y + index * line_height
        draw.text((x, y), line, font=font, fill=TEXT_COLOR, stroke_width=STROKE_WIDTH, stroke_fill=STROKE_COLOR)


def render_thumbnail(
    source_image_path: str,
    plan: ThumbnailPlan,
    output_path: str,
    font_path: Optional[str] = None,
) -> None:
    """Deterministically render a 1280x720 thumbnail: center-crop/resize
    the source image (never stretched), composite a translucent contrast
    panel, then draw ``plan.hook_text`` using a fixed layout rule derived
    from ``plan.composition`` - never arbitrary LLM-provided coordinates.

    ``output_path``'s directory is created if missing. Never overwrites
    ``source_image_path``.

    Raises:
        ThumbnailRenderError: If the source image can't be opened/processed,
            or the output file can't be written.
    """
    try:
        with Image.open(source_image_path) as source:
            base = _crop_to_aspect(source.convert("RGB"), THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
    except Exception as e:
        raise ThumbnailRenderError(f"Failed to open/process source image: {e}") from e

    layout = _layout_for_composition(plan.composition)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(layout["panel"], fill=(*OVERLAY_COLOR, OVERLAY_OPACITY))
    canvas = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = layout["text_box"]
    font, lines = _fit_text(draw, plan.hook_text, x1 - x0, y1 - y0, font_path)
    if lines:
        _draw_lines(draw, lines, font, layout)

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        canvas.save(output_path, quality=92)
    except Exception as e:
        raise ThumbnailRenderError(f"Failed to write thumbnail output: {e}") from e
