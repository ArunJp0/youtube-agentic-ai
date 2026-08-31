# Deterministic representative-frame sampling for Visual QC.
#
# Never inspects every frame of a clip: picks a small, fixed number of
# timestamps based only on the clip's own duration, so Visual QC's vision
# calls stay cheap regardless of how long a selected clip happens to be.
from __future__ import annotations

from typing import List

# Below this duration, a single mid-point frame is representative enough -
# slicing a very short clip into multiple samples adds vision-call cost
# without adding useful information.
SHORT_CLIP_MAX_SECONDS = 6.0

# Between SHORT_CLIP_MAX_SECONDS and this, two samples (25%/75%) are enough
# to catch a clip that changes partway through. Beyond this, three samples
# (25%/50%/75%) are used - still a small, fixed cap, never proportional to
# duration.
MEDIUM_CLIP_MAX_SECONDS = 15.0


def calculate_sample_timestamps(duration_seconds: float) -> List[float]:
    """Deterministically choose representative-frame timestamps (in
    seconds) for a clip of the given duration.

    Args:
        duration_seconds: The clip's own duration

    Returns:
        A short, fixed-size list of timestamps (1-3 entries), ordered
        earliest first
    """
    if duration_seconds <= 0:
        return [0.0]
    if duration_seconds <= SHORT_CLIP_MAX_SECONDS:
        fractions = [0.5]
    elif duration_seconds <= MEDIUM_CLIP_MAX_SECONDS:
        fractions = [0.25, 0.75]
    else:
        fractions = [0.25, 0.5, 0.75]
    return [round(duration_seconds * fraction, 3) for fraction in fractions]
