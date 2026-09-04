# BGM Catalog

Curated, license-known background music tracks for `AudioMixingService`
(`src/services/audio_mixing_service.py`). This is the **only** source of
music the system will ever select from - nothing is downloaded, scraped,
or guessed at runtime.

## Copyright / source policy

- Only use tracks from an approved source with known usage rights.
- Preferred source for the MVP: the [YouTube Audio Library](https://www.youtube.com/audiolibrary),
  preferring tracks marked **"Attribution not required"**.
- Prefer instrumental/background tracks (no vocals/lyrics).
- Do **not** download from "No Copyright Music" YouTube channels or any
  source whose actual license terms aren't known.

## Adding a track

1. Obtain the track from an approved source (see above).
2. Save the audio file under `assets/bgm/tracks/`, e.g.
   `assets/bgm/tracks/calm-piano-01.mp3`.
3. Add one entry to `catalog.json` (a JSON array) describing it:

```json
{
  "track_id": "calm-piano-01",
  "file_path": "tracks/calm-piano-01.mp3",
  "title": "Track title, as credited by the source",
  "source": "YouTube Audio Library",
  "license_type": "youtube_audio_library_no_attribution",
  "attribution_required": false,
  "attribution_text": null,
  "genre": "ambient",
  "mood_tags": ["calm", "thoughtful", "subtle"],
  "energy_level": "low",
  "instrumental": true,
  "duration_seconds": 128.0
}
```

Notes:

- `file_path` is resolved relative to this directory (`assets/bgm/`)
  unless given as an absolute path.
- `energy_level` must be `"low"`, `"medium"`, or `"high"`.
- `instrumental` should be `true` for narrated-video use - vocal tracks
  are excluded by default during selection.
- `license_type`/`source`/`attribution_required`/`attribution_text` must
  accurately reflect the track's real origin - never mark a track
  approved unless its license is actually known.

Once at least one track is in the catalog, run the standalone demo:

```
python -m src.bgm_demo "<topic>"
```
