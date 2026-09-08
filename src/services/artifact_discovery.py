# Shared helpers for locating the most recently produced pipeline output
# artifacts on disk, for standalone demos that validate against an
# already-completed real run instead of regenerating one.
#
# src/bgm_demo.py, src/metadata_demo.py, and src/thumbnail_demo.py each
# already contain their own private, near-identical copy of the final-video
# lookup (predating this module) - left untouched here, since they are
# working, already-approved standalone milestones and this module's own
# instructions are to avoid modifying them unless required. New callers
# (starting with src/compliance_demo.py) should use this module instead of
# adding yet another private copy.
from __future__ import annotations

import glob
import os
from typing import Optional

# Preference order: the final user-facing output first, falling back
# toward earlier pipeline stages' outputs if later ones don't exist yet.
_VIDEO_SUFFIXES_BY_PREFERENCE = ("-captioned-bgm.mp4", "-captioned.mp4", ".mp4")


def find_latest_final_video(directory: str) -> Optional[str]:
    """Prefer the final BGM-mixed MP4; fall back to the captioned MP4, then
    any plain assembled MP4."""
    if not os.path.isdir(directory):
        return None
    for suffix in _VIDEO_SUFFIXES_BY_PREFERENCE:
        if suffix == ".mp4":
            candidates = [
                path
                for path in glob.glob(os.path.join(directory, "*.mp4"))
                if not path.endswith("-captioned.mp4") and not path.endswith("-captioned-bgm.mp4")
            ]
        else:
            candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None


def find_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """Most recently modified file in ``directory`` matching ``pattern``, or
    None if the directory doesn't exist or has no matches."""
    if not os.path.isdir(directory):
        return None
    candidates = glob.glob(os.path.join(directory, pattern))
    return max(candidates, key=os.path.getmtime) if candidates else None
