# Lightweight sequence-level repetition QC for the final selected-asset
# timeline. Deliberately asset-ID/position based - not frame-level computer
# vision duplicate detection, which is out of scope for this milestone.
from __future__ import annotations

from typing import Dict, List


def detect_repetition_warnings(asset_ids_in_order: List[str], min_gap: int) -> List[str]:
    """Flag any asset id that reappears too soon after its previous use.

    Controlled reuse in long videos is acceptable (see
    VisualMediaService's own reuse-after-gap policy) - this only produces
    human-readable warnings, it never rejects or mutates anything.

    Args:
        asset_ids_in_order: The final, ordered sequence of selected asset
            ids across the whole video (one per visual slot)
        min_gap: Reuse within fewer than this many slots of the same
            asset's previous occurrence is flagged (a gap of exactly 1
            means immediate back-to-back repetition)

    Returns:
        Human-readable warning strings, one per flagged reuse - empty if
        no problematic repetition was found
    """
    warnings: List[str] = []
    last_seen_position: Dict[str, int] = {}

    for position, asset_id in enumerate(asset_ids_in_order):
        if not asset_id:
            continue
        previous_position = last_seen_position.get(asset_id)
        if previous_position is not None:
            gap = position - previous_position
            if gap == 1:
                warnings.append(f"Asset {asset_id!r} reused back-to-back at slot {position + 1}")
            elif gap <= min_gap:
                warnings.append(
                    f"Asset {asset_id!r} reused after only {gap} slot(s) at slot {position + 1}"
                )
        last_seen_position[asset_id] = position

    return warnings
