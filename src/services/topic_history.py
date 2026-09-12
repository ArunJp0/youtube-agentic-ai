# Reads real produced-run topic history for Topic Planner duplicate
# prevention - the ground truth "already generated" source, reusing the
# existing ProvenanceManifestStore (one JSON per real completed run,
# already carrying topic + created_at) rather than inventing a parallel
# history log. Read-only: never writes, never modifies provenance_store.py.
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

from src.services.provenance_store import (
    DEFAULT_PROVENANCE_OUTPUT_DIR,
    ProvenanceManifestStore,
    ProvenanceStoreError,
)


def load_recent_produced_topics(
    provenance_dir: str = DEFAULT_PROVENANCE_OUTPUT_DIR,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Return ``(topic, created_at)`` pairs for every real completed run
    with a persisted ProvenanceManifest, most-recent first.

    A corrupt/unreadable manifest is skipped (never raised) - duplicate
    prevention degrades to "one less known topic" rather than failing the
    whole planning run over one bad file.

    Args:
        provenance_dir: Directory ProvenanceManifestStore reads from
        limit: Cap the number of most-recent entries returned; None for all

    Returns:
        List of (topic, created_at) tuples, most-recent first.
    """
    if not os.path.isdir(provenance_dir):
        return []

    store = ProvenanceManifestStore(provenance_dir)
    entries: List[Tuple[str, str]] = []
    for path in glob.glob(os.path.join(provenance_dir, "*.json")):
        run_id = os.path.splitext(os.path.basename(path))[0]
        try:
            manifest = store.read(run_id)
        except ProvenanceStoreError:
            continue
        if manifest is not None:
            entries.append((manifest.topic, manifest.created_at))

    entries.sort(key=lambda item: item[1], reverse=True)
    return entries[:limit] if limit is not None else entries
