# Provenance manifest persistence boundary - the ONLY place that reads or
# writes provenance manifest files. Deliberately narrow: no agent writes its
# own provenance file directly; everything goes through this store, so the
# on-disk format/location is a single, auditable decision.
#
# Writes are atomic (temp file in the same directory, then os.replace) so an
# interrupted process can never leave a partially-written/corrupt manifest
# behind - a reader either sees the complete previous file or the complete
# new one, never a half-written one.
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from pydantic import ValidationError

from src.models.provenance import ProvenanceManifest
from src.services.script_context_reconstruction import original_base_name

DEFAULT_PROVENANCE_OUTPUT_DIR = os.path.join("output", "provenance")


class ProvenanceStoreError(Exception):
    """Raised when a provenance manifest file exists but cannot be read or
    parsed (corrupt/malformed) - distinct from a manifest simply not
    existing (which read/find_for_video report as ``None``, never an
    error), so callers can tell "no manifest" apart from "broken manifest"."""


class ProvenanceManifestStore:
    """Reads/writes ``ProvenanceManifest`` records under a local directory,
    one JSON file per run, named after the run's own stable identifier."""

    def __init__(self, output_dir: str = DEFAULT_PROVENANCE_OUTPUT_DIR) -> None:
        self.output_dir = output_dir

    def write(self, manifest: ProvenanceManifest) -> str:
        """Atomically write ``manifest`` to ``<output_dir>/<run_id>.json``,
        overwriting any existing manifest for the same run_id.

        Returns:
            The path written.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        final_path = os.path.join(self.output_dir, f"{manifest.run_id}.json")

        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, prefix=".tmp-provenance-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, final_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return final_path

    def read(self, run_id: str) -> Optional[ProvenanceManifest]:
        """Read the manifest for ``run_id``, or ``None`` if it doesn't exist.

        Raises:
            ProvenanceStoreError: If a file exists at that path but is
                corrupt/unparseable - never silently treated as "missing".
        """
        return self._read_path(os.path.join(self.output_dir, f"{run_id}.json"))

    def find_for_video(self, video_path: str) -> Optional[ProvenanceManifest]:
        """Locate the manifest matching ``video_path``'s own run identifier
        (derived the same way CaptionService/AudioMixingService/MetadataAgent
        already recover a video's base slug from any of its
        original/-captioned/-captioned-bgm variants - see
        ``script_context_reconstruction.original_base_name``), or ``None``
        if this video predates provenance persistence (a legacy artifact -
        valid, never an error, never fabricated).
        """
        return self.read(original_base_name(video_path))

    def _read_path(self, path: str) -> Optional[ProvenanceManifest]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProvenanceManifest.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            raise ProvenanceStoreError(f"Provenance manifest at '{path}' is corrupt/unreadable: {e}") from e
