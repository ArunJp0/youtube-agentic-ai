# Tests for PexelsMediaProvider. All httpx calls are mocked in every test
# in this module, so no real network call is ever made.
from __future__ import annotations

import httpx
import pytest

from src.tools.media_provider import MediaProviderError
from src.tools.pexels_media_provider import PexelsMediaProvider


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, json_data=None, content: bytes = b"", status_code: int = 200) -> None:
        self._json_data = json_data or {}
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://api.pexels.com/fake")
            raise httpx.HTTPStatusError(
                f"HTTP error {self.status_code}", request=request, response=self  # type: ignore[arg-type]
            )

    def json(self):
        return self._json_data


class FakeAsyncClient:
    """Stand-in for httpx.AsyncClient used as an async context manager."""

    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.last_url = None
        self.last_params = None
        self.last_headers = None

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None, headers=None):
        self.last_url = url
        self.last_params = params
        self.last_headers = headers
        if self.exc is not None:
            raise self.exc
        return self.response


def _video_payload(videos):
    return {"videos": videos}


def _photo_payload(photos):
    return {"photos": photos}


class TestPexelsMediaProviderInit:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(MediaProviderError, match="PEXELS_API_KEY"):
            PexelsMediaProvider(api_key=None)

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(MediaProviderError, match="PEXELS_API_KEY"):
            PexelsMediaProvider(api_key="")

    def test_name(self) -> None:
        assert PexelsMediaProvider(api_key="k").name == "pexels"


class TestPexelsMediaProviderSearch:
    @pytest.mark.asyncio
    async def test_video_search_returns_candidates(self, monkeypatch) -> None:
        payload = _video_payload(
            [
                {
                    "id": 4567890,
                    "url": "https://www.pexels.com/video/123",
                    "duration": 10,
                    "user": {"name": "Jane Doe"},
                    "video_files": [
                        {"link": "https://cdn.pexels.com/vid_sd.mp4", "width": 640, "height": 360, "quality": "sd", "file_type": "video/mp4"},
                        {"link": "https://cdn.pexels.com/vid_hd.mp4", "width": 1920, "height": 1080, "quality": "hd", "file_type": "video/mp4"},
                    ],
                }
            ]
        )
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        candidates = await provider.search("ocean", prefer_video=True, max_results=5)

        assert len(candidates) == 1
        assert candidates[0].asset_type == "video"
        assert candidates[0].download_url == "https://cdn.pexels.com/vid_hd.mp4"  # hd preferred over sd
        assert candidates[0].attribution == "Jane Doe"
        assert candidates[0].duration_seconds == 10
        assert candidates[0].provider_asset_id == "4567890"
        assert fake_client.last_params["orientation"] == "landscape"
        assert fake_client.last_headers["Authorization"] == "test-key"

    @pytest.mark.asyncio
    async def test_video_missing_id_leaves_provider_asset_id_none(self, monkeypatch) -> None:
        payload = _video_payload(
            [
                {
                    "url": "https://www.pexels.com/video/123",
                    "video_files": [
                        {"link": "https://cdn.pexels.com/vid.mp4", "file_type": "video/mp4", "quality": "hd"}
                    ],
                }
            ]
        )
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        candidates = await provider.search("ocean")
        assert candidates[0].provider_asset_id is None

    @pytest.mark.asyncio
    async def test_falls_back_to_photos_when_no_videos(self, monkeypatch) -> None:
        calls = {"n": 0}
        video_response = FakeResponse(_video_payload([]))
        photo_response = FakeResponse(
            _photo_payload(
                [
                    {
                        "url": "https://www.pexels.com/photo/456",
                        "photographer": "John Smith",
                        "width": 1920,
                        "height": 1080,
                        "src": {"large2x": "https://cdn.pexels.com/photo_large2x.jpg"},
                    }
                ]
            )
        )

        class SwitchingClient(FakeAsyncClient):
            async def get(self, url, params=None, headers=None):
                calls["n"] += 1
                self.response = video_response if "videos" in url else photo_response
                return await super().get(url, params=params, headers=headers)

        fake_client = SwitchingClient()
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        candidates = await provider.search("mountains", prefer_video=True, max_results=5)

        assert len(candidates) == 1
        assert candidates[0].asset_type == "image"
        assert candidates[0].download_url == "https://cdn.pexels.com/photo_large2x.jpg"
        assert candidates[0].attribution == "John Smith"
        assert calls["n"] == 2  # video search, then photo fallback

    @pytest.mark.asyncio
    async def test_prefer_video_false_skips_video_search(self, monkeypatch) -> None:
        payload = _photo_payload(
            [
                {
                    "url": "https://www.pexels.com/photo/789",
                    "photographer": "A",
                    "src": {"original": "https://cdn.pexels.com/photo.jpg"},
                }
            ]
        )
        fake_client = FakeAsyncClient(response=FakeResponse(payload))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        candidates = await provider.search("forest", prefer_video=False)

        assert len(candidates) == 1
        assert candidates[0].asset_type == "image"
        assert "v1/search" in fake_client.last_url

    @pytest.mark.asyncio
    async def test_video_with_no_mp4_files_is_skipped(self, monkeypatch) -> None:
        payload = _video_payload(
            [{"url": "https://www.pexels.com/video/1", "video_files": []}]
        )
        fake_client = FakeAsyncClient(response=FakeResponse(_video_payload([])))

        class SwitchingClient(FakeAsyncClient):
            async def get(self, url, params=None, headers=None):
                self.response = FakeResponse(payload) if "videos" in url else FakeResponse(_photo_payload([]))
                return await super().get(url, params=params, headers=headers)

        fake_client = SwitchingClient()
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        candidates = await provider.search("empty case", prefer_video=True)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_http_error_raises_provider_error(self, monkeypatch) -> None:
        fake_client = FakeAsyncClient(response=FakeResponse({}, status_code=403))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="bad-key")
        with pytest.raises(MediaProviderError, match="403"):
            await provider.search("ocean")

    @pytest.mark.asyncio
    async def test_network_error_raises_provider_error(self, monkeypatch) -> None:
        fake_client = FakeAsyncClient(exc=httpx.ConnectError("connection refused"))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        with pytest.raises(MediaProviderError):
            await provider.search("ocean")


class TestPexelsMediaProviderDownload:
    @pytest.mark.asyncio
    async def test_download_writes_file(self, monkeypatch, tmp_path) -> None:
        fake_client = FakeAsyncClient(response=FakeResponse(content=b"FAKEBYTES"))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        from src.tools.media_provider import MediaCandidate

        candidate = MediaCandidate(
            asset_type="image", download_url="https://cdn.pexels.com/x.jpg", source_url="https://pexels.com/x"
        )
        output_path = str(tmp_path / "x.jpg")
        await provider.download(candidate, output_path)

        with open(output_path, "rb") as f:
            assert f.read() == b"FAKEBYTES"

    @pytest.mark.asyncio
    async def test_download_http_error_raises(self, monkeypatch, tmp_path) -> None:
        fake_client = FakeAsyncClient(response=FakeResponse({}, status_code=404))
        monkeypatch.setattr(httpx, "AsyncClient", fake_client)

        provider = PexelsMediaProvider(api_key="test-key")
        from src.tools.media_provider import MediaCandidate

        candidate = MediaCandidate(
            asset_type="image", download_url="https://cdn.pexels.com/gone.jpg", source_url="https://pexels.com/gone"
        )
        with pytest.raises(MediaProviderError, match="404"):
            await provider.download(candidate, str(tmp_path / "gone.jpg"))
