"""Tests for the Flask Playlist Maker app."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from app import (
    CandidateTrack,
    CuratedPlaylist,
    SearchQueries,
    SpotifyClient,
    _is_rate_limited,
    _rate_limit_store,
    app,
    build_prompt,
    discover_candidates,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Reset rate limit store between tests."""
    _rate_limit_store.clear()
    yield
    _rate_limit_store.clear()


# --- CSRF Protection ---

class TestCSRF:
    def test_rejects_no_origin_no_referer(self, client):
        resp = client.post("/api/generate", json={"prompt": "test"})
        assert resp.status_code == 403
        assert "Invalid request origin" in resp.get_json()["error"]

    def test_rejects_wrong_origin(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code == 403

    def test_rejects_wrong_referer(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
            headers={"Referer": "https://evil.com/page"},
        )
        assert resp.status_code == 403

    def test_accepts_matching_origin(self, client):
        # Should pass CSRF but fail later (no Spotify mock) — just verify not 403
        with patch("app.get_spotify"), patch("app.generate_search_queries", return_value=["q"]), \
             patch("app.discover_candidates", return_value=[]), patch("app._is_rate_limited", return_value=False):
            resp = client.post(
                "/api/generate",
                json={"prompt": "test"},
                headers={"Origin": "http://localhost"},
            )
            assert resp.status_code != 403


# --- Rate Limiting ---

class TestRateLimiting:
    def test_allows_under_limit(self):
        for _ in range(5):
            assert not _is_rate_limited("1.2.3.4")

    def test_blocks_over_limit(self):
        for _ in range(5):
            _is_rate_limited("1.2.3.4")
        assert _is_rate_limited("1.2.3.4")

    def test_different_ips_independent(self):
        for _ in range(5):
            _is_rate_limited("1.1.1.1")
        assert _is_rate_limited("1.1.1.1")
        assert not _is_rate_limited("2.2.2.2")

    def test_window_expires(self):
        with patch("app.time.time") as mock_time:
            mock_time.return_value = 1000.0
            for _ in range(5):
                _is_rate_limited("5.5.5.5")
            assert _is_rate_limited("5.5.5.5")

            # Advance past the window
            mock_time.return_value = 1061.0
            assert not _is_rate_limited("5.5.5.5")

    def test_rate_limit_returns_429(self, client):
        with patch("app._is_rate_limited", return_value=True):
            resp = client.post(
                "/api/generate",
                json={"prompt": "test"},
                headers={"Origin": "http://localhost"},
            )
            assert resp.status_code == 429
            assert "Too many requests" in resp.get_json()["error"]


# --- Input Validation ---

class TestInputValidation:
    @pytest.fixture
    def headers(self):
        return {"Origin": "http://localhost"}

    def test_missing_prompt(self, client, headers):
        resp = client.post("/api/generate", json={}, headers=headers)
        assert resp.status_code == 400

    def test_empty_prompt(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "   "}, headers=headers)
        assert resp.status_code == 400

    def test_prompt_too_long(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "x" * 501}, headers=headers)
        assert resp.status_code == 400
        assert "too long" in resp.get_json()["error"]

    def test_invalid_energy(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "test", "energy": 6}, headers=headers)
        assert resp.status_code == 400

    def test_invalid_mood(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "test", "moods": ["fake"]}, headers=headers)
        assert resp.status_code == 400

    def test_too_many_moods(self, client, headers):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "moods": ["happy", "dreamy", "peaceful", "romantic"]},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_invalid_context(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "test", "context": "skydiving"}, headers=headers)
        assert resp.status_code == 400

    def test_invalid_era_range(self, client, headers):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "era_from": 2020, "era_to": 2010},
            headers=headers,
        )
        assert resp.status_code == 400


# --- build_prompt ---

class TestBuildPrompt:
    def test_basic_prompt(self):
        assert build_prompt("chill vibes") == "chill vibes"

    def test_with_energy(self):
        result = build_prompt("test", energy=5)
        assert "very high energy" in result

    def test_with_moods(self):
        result = build_prompt("test", moods=["happy", "dreamy"])
        assert "happy" in result and "dreamy" in result

    def test_with_context(self):
        result = build_prompt("test", context="working out")
        assert "working out" in result

    def test_with_era(self):
        result = build_prompt("test", era_from=1980, era_to=1989)
        assert "1980-1989" in result

    def test_with_seed(self):
        result = build_prompt("test", seed="Khruangbin")
        assert "Khruangbin" in result


# --- SpotifyClient ---

@pytest.fixture
def spotify_client():
    with patch("app.SpotifyOAuth"):
        client = SpotifyClient.__new__(SpotifyClient)
        client.auth = MagicMock()
        client.auth.get_access_token = MagicMock(return_value="fake-token")
        client.session = MagicMock()
        client._token_lock = MagicMock()
        return client


class TestSpotifyClient:
    def test_headers(self, spotify_client):
        headers = spotify_client._headers()
        assert headers["Authorization"] == "Bearer fake-token"

    def test_successful_request(self, spotify_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"ok": true}'
        mock_resp.json.return_value = {"ok": True}
        spotify_client.session.request = MagicMock(return_value=mock_resp)

        result = spotify_client._request("GET", "me")
        assert result == {"ok": True}

    @patch("app.time.sleep")
    def test_429_retry(self, mock_sleep, spotify_client):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "2"}

        success = MagicMock()
        success.status_code = 200
        success.content = b'{"ok": true}'
        success.json.return_value = {"ok": True}

        spotify_client.session.request = MagicMock(side_effect=[rate_limited, success])
        result = spotify_client._request("GET", "search")
        assert result == {"ok": True}
        mock_sleep.assert_called_once_with(2)

    def test_4xx_raises(self, spotify_client):
        error_resp = MagicMock()
        error_resp.status_code = 403
        error_resp.raise_for_status.side_effect = requests.HTTPError(response=error_resp)
        spotify_client.session.request = MagicMock(return_value=error_resp)

        with pytest.raises(requests.HTTPError):
            spotify_client._request("GET", "me")

    def test_search_tracks(self, spotify_client):
        spotify_client._request = MagicMock(return_value={
            "tracks": {"items": [
                {"name": "Song A", "artists": [{"name": "Artist 1"}], "uri": "spotify:track:1"},
            ]}
        })
        results = spotify_client.search_tracks("test")
        assert len(results) == 1
        assert results[0].title == "Song A"

    def test_create_playlist(self, spotify_client):
        call_log = []

        def mock_request(method, path, **kwargs):
            call_log.append((method, path))
            if path == "me/playlists":
                return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}
            return {}

        spotify_client._request = MagicMock(side_effect=mock_request)
        url, count = spotify_client.create_playlist("Test", "Desc", ["uri:1", "uri:2"])
        assert url == "https://open.spotify.com/playlist/pl1"
        assert count == 2


# --- discover_candidates ---

class TestDiscoverCandidates:
    def test_deduplicates(self, spotify_client):
        track = CandidateTrack(title="Song", artist="Artist", uri="spotify:track:1")
        spotify_client.search_tracks = MagicMock(return_value=[track])

        candidates = discover_candidates(spotify_client, ["q1", "q2"], max_workers=2)
        assert len(candidates) == 1

    def test_era_doubles_queries(self, spotify_client):
        spotify_client.search_tracks = MagicMock(return_value=[])
        discover_candidates(spotify_client, ["q1", "q2"], max_workers=1, era_from=1980, era_to=1989)
        # Should search with 4 queries: 2 filtered + 2 unfiltered
        assert spotify_client.search_tracks.call_count == 4

    def test_handles_search_failure(self, spotify_client):
        spotify_client.search_tracks = MagicMock(side_effect=requests.HTTPError())
        candidates = discover_candidates(spotify_client, ["q1"], max_workers=1)
        assert candidates == []


# --- Full generate endpoint (mocked) ---

class TestGenerateEndpoint:
    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    @patch("app.curate_playlist")
    def test_success(self, mock_curate, mock_discover, mock_queries, mock_spotify, client):
        mock_queries.return_value = ["q1", "q2"]
        mock_discover.return_value = [
            CandidateTrack(title="Song A", artist="Artist 1", uri="spotify:track:1"),
            CandidateTrack(title="Song B", artist="Artist 2", uri="spotify:track:2"),
        ]
        mock_curate.return_value = CuratedPlaylist(
            playlist_name="Chill Vibes",
            description="Relaxing",
            selected_uris=["spotify:track:1", "spotify:track:2"],
        )
        mock_sp = MagicMock()
        mock_sp.create_playlist.return_value = ("https://open.spotify.com/playlist/abc", 2)
        mock_spotify.return_value = mock_sp

        resp = client.post(
            "/api/generate",
            json={"prompt": "chill vibes", "count": 2},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["playlist_name"] == "Chill Vibes"
        assert data["playlist_url"] == "https://open.spotify.com/playlist/abc"
        assert len(data["tracks"]) == 2

    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    def test_no_candidates(self, mock_discover, mock_queries, mock_spotify, client):
        mock_queries.return_value = ["q1"]
        mock_discover.return_value = []
        mock_spotify.return_value = MagicMock()

        resp = client.post(
            "/api/generate",
            json={"prompt": "obscure request"},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 404
        assert "No tracks found" in resp.get_json()["error"]
