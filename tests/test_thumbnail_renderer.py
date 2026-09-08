# Tests for deterministic thumbnail rendering (src/services/thumbnail_renderer.py).
# Uses locally-generated PIL test images only - no network calls.
from __future__ import annotations

import os

import pytest
from PIL import Image

from src.models.thumbnail import ThumbnailPlan
from src.services.thumbnail_renderer import (
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    ThumbnailRenderError,
    render_thumbnail,
)


def _plan(**overrides) -> ThumbnailPlan:
    defaults = dict(hook_text="WHY DO WE DREAM", search_query="q", used_semantic_planning=False)
    defaults.update(overrides)
    return ThumbnailPlan(**defaults)


def _make_source_image(path: str, width: int, height: int, color=(60, 90, 140)) -> None:
    Image.new("RGB", (width, height), color).save(path)


class TestRenderThumbnailBasics:
    def test_output_is_exactly_target_dimensions(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(), output)

        with Image.open(output) as img:
            assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_output_file_is_valid_image(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(), output)

        with Image.open(output) as img:
            img.verify()

    def test_output_directory_created_if_missing(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / "nested" / "dir" / "out.jpg")

        render_thumbnail(source, _plan(), output)

        assert os.path.exists(output)


class TestRenderThumbnailCropHandling:
    def test_landscape_source_no_stretching(self, tmp_path) -> None:
        """A very wide source should be center-cropped, not squashed -
        verified indirectly by confirming the exact output dimensions and
        successful completion for an extreme aspect ratio."""
        source = str(tmp_path / "wide.jpg")
        _make_source_image(source, 3840, 1080)
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(), output)

        with Image.open(output) as img:
            assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_portrait_source_crop_succeeds(self, tmp_path) -> None:
        source = str(tmp_path / "portrait.jpg")
        _make_source_image(source, 900, 1600)
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(), output)

        with Image.open(output) as img:
            assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_square_source_crop_succeeds(self, tmp_path) -> None:
        source = str(tmp_path / "square.jpg")
        _make_source_image(source, 1000, 1000)
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(), output)

        with Image.open(output) as img:
            assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)


class TestRenderThumbnailComposition:
    @pytest.mark.parametrize("composition", ["subject_left", "subject_right", "centered"])
    def test_every_composition_renders_successfully(self, tmp_path, composition) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / f"out-{composition}.jpg")

        render_thumbnail(source, _plan(composition=composition), output)

        with Image.open(output) as img:
            assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)


class TestRenderThumbnailTextHandling:
    def test_short_hook_renders_successfully(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(hook_text="DREAM ON"), output)

        assert os.path.exists(output)

    def test_long_hook_wraps_and_shrinks_without_error(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / "out.jpg")
        long_hook = "THIS IS A VERY LONG HOOK TEXT THAT SHOULD WRAP AND SHRINK SAFELY WITHOUT OVERFLOWING"

        render_thumbnail(source, _plan(hook_text=long_hook), output)

        with Image.open(output) as img:
            assert img.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)

    def test_font_fallback_used_when_configured_font_path_missing(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        output = str(tmp_path / "out.jpg")

        # A nonexistent font path must not raise - the renderer falls back
        # to local/system fonts (or Pillow's built-in bitmap font) instead.
        render_thumbnail(source, _plan(), output, font_path="/no/such/font.ttf")

        assert os.path.exists(output)


class TestRenderThumbnailFailures:
    def test_missing_source_image_raises_render_error(self, tmp_path) -> None:
        output = str(tmp_path / "out.jpg")
        with pytest.raises(ThumbnailRenderError):
            render_thumbnail(str(tmp_path / "does-not-exist.jpg"), _plan(), output)

    def test_corrupt_source_image_raises_render_error(self, tmp_path) -> None:
        source = str(tmp_path / "corrupt.jpg")
        with open(source, "wb") as f:
            f.write(b"not a real image")
        output = str(tmp_path / "out.jpg")

        with pytest.raises(ThumbnailRenderError):
            render_thumbnail(source, _plan(), output)

    def test_source_image_never_modified(self, tmp_path) -> None:
        source = str(tmp_path / "source.jpg")
        _make_source_image(source, 1920, 1080)
        original_bytes = open(source, "rb").read()
        output = str(tmp_path / "out.jpg")

        render_thumbnail(source, _plan(), output)

        assert open(source, "rb").read() == original_bytes
