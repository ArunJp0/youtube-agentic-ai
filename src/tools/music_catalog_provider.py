# BGM catalog provider abstraction: a curated, license-known local music
# library. AudioMixingService/MusicSelectionService only ever depend on
# this interface - never on a filesystem/JSON format directly - so another
# approved provider (e.g. a licensed Mixkit catalog) could be added later
# without changing either service.
#
# Deliberately narrow: this provider only ever returns tracks a human has
# already vetted and placed in the catalog. It never searches, scrapes, or
# downloads music from the internet - that would defeat the whole point of
# a copyright-safe MVP source.
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.music import BGMTrack

DEFAULT_CATALOG_PATH = os.path.join("assets", "bgm", "catalog.json")


class MusicCatalogProviderError(Exception):
    """Raised when the approved BGM catalog cannot be found, read, or parsed."""


class MusicCatalogProvider(ABC):
    """Abstract base class for approved BGM catalog sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. 'mock', 'local'."""
        raise NotImplementedError

    @abstractmethod
    def list_tracks(self) -> List[BGMTrack]:
        """Return every track in the approved catalog, in catalog order.

        Never invents, searches for, or downloads a track - only returns
        what the catalog already explicitly declares.

        Raises:
            MusicCatalogProviderError: If the catalog cannot be loaded/parsed.
        """
        raise NotImplementedError


class LocalMusicCatalogProvider(MusicCatalogProvider):
    """Reads a curated, license-known BGM catalog from a local JSON file.

    The catalog is a manually maintained list of tracks the user has
    legally obtained and licensed appropriately (e.g. YouTube Audio Library
    tracks marked "Attribution not required"). See assets/bgm/README.md for
    the exact file format. An empty or missing catalog is not itself an
    error here - MusicSelectionService/AudioMixingService decide how to
    react to "no eligible track" - but a malformed catalog file is.
    """

    def __init__(self, catalog_path: str = DEFAULT_CATALOG_PATH) -> None:
        self.catalog_path = catalog_path

    @property
    def name(self) -> str:
        return "local"

    def list_tracks(self) -> List[BGMTrack]:
        if not os.path.exists(self.catalog_path):
            raise MusicCatalogProviderError(
                f"BGM catalog file not found: {self.catalog_path}. "
                "See assets/bgm/README.md for the required catalog format."
            )
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise MusicCatalogProviderError(f"Failed to read BGM catalog '{self.catalog_path}': {e}") from e

        if not isinstance(raw, list):
            raise MusicCatalogProviderError("BGM catalog must be a JSON array of track entries")

        catalog_dir = os.path.dirname(os.path.abspath(self.catalog_path))
        tracks: List[BGMTrack] = []
        for entry in raw:
            try:
                track = BGMTrack.model_validate(entry)
            except Exception as e:
                raise MusicCatalogProviderError(f"Invalid BGM catalog entry {entry!r}: {e}") from e
            if not os.path.isabs(track.file_path):
                track = track.model_copy(
                    update={"file_path": os.path.normpath(os.path.join(catalog_dir, track.file_path))}
                )
            tracks.append(track)
        return tracks


class MockMusicCatalogProvider(MusicCatalogProvider):
    """In-memory catalog for tests - no filesystem/JSON involved."""

    def __init__(self, tracks: Optional[List[BGMTrack]] = None) -> None:
        self.tracks = tracks if tracks is not None else []

    @property
    def name(self) -> str:
        return "mock"

    def list_tracks(self) -> List[BGMTrack]:
        return list(self.tracks)
