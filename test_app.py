"""Tests for the Flask Playlist Maker app."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from app import (
    API_KEY,
    CandidateTrack,
    CuratedPlaylist,
    REQUEST_DEADLINE,
    SearchQueries,
    SpotifyClient,
    _check_deadline,
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

    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    @patch("app.curate_playlist")
    def test_warning_when_fewer_tracks(self, mock_curate, mock_discover, mock_queries, mock_spotify, client):
        """Issue #25: warning field when fewer tracks than requested."""
        mock_queries.return_value = ["q1"]
        mock_discover.return_value = [
            CandidateTrack(title="Song A", artist="Artist 1", uri="spotify:track:1"),
        ]
        mock_curate.return_value = CuratedPlaylist(
            playlist_name="Test", description="Desc",
            selected_uris=["spotify:track:1"],
        )
        mock_sp = MagicMock()
        mock_sp.create_playlist.return_value = ("https://open.spotify.com/playlist/abc", 1)
        mock_spotify.return_value = mock_sp

        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "count": 10},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warning" in data
        assert "1 track" in data["warning"]
        assert data["tracks_found"] == 1

    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    @patch("app.curate_playlist")
    def test_no_warning_when_enough_tracks(self, mock_curate, mock_discover, mock_queries, mock_spotify, client):
        """No warning when track count meets requested count."""
        tracks = [
            CandidateTrack(title=f"Song {i}", artist=f"Artist {i}", uri=f"spotify:track:{i}")
            for i in range(2)
        ]
        mock_queries.return_value = ["q1"]
        mock_discover.return_value = tracks
        mock_curate.return_value = CuratedPlaylist(
            playlist_name="Test", description="Desc",
            selected_uris=[t.uri for t in tracks],
        )
        mock_sp = MagicMock()
        mock_sp.create_playlist.return_value = ("https://open.spotify.com/playlist/abc", 2)
        mock_spotify.return_value = mock_sp

        resp = client.post(
            "/api/generate",
            json={"prompt": "test", "count": 2},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 200
        assert "warning" not in resp.get_json()


# --- Issue #21: search_tracks missing fields ---

class TestSearchTracksMissingFields:
    def test_skips_items_without_name(self, spotify_client):
        spotify_client._request = MagicMock(return_value={
            "tracks": {"items": [
                {"artists": [{"name": "Artist"}], "uri": "spotify:track:1"},
                {"name": "Good Song", "artists": [{"name": "Artist"}], "uri": "spotify:track:2"},
            ]}
        })
        results = spotify_client.search_tracks("test")
        assert len(results) == 1
        assert results[0].title == "Good Song"

    def test_skips_items_without_uri(self, spotify_client):
        spotify_client._request = MagicMock(return_value={
            "tracks": {"items": [
                {"name": "No URI Song", "artists": [{"name": "Artist"}]},
                {"name": "Good Song", "artists": [{"name": "Artist"}], "uri": "spotify:track:2"},
            ]}
        })
        results = spotify_client.search_tracks("test")
        assert len(results) == 1
        assert results[0].uri == "spotify:track:2"

    def test_skips_items_with_empty_name(self, spotify_client):
        spotify_client._request = MagicMock(return_value={
            "tracks": {"items": [
                {"name": "", "artists": [{"name": "Artist"}], "uri": "spotify:track:1"},
            ]}
        })
        results = spotify_client.search_tracks("test")
        assert len(results) == 0


# --- Issue #24: create_playlist malformed response ---

class TestCreatePlaylistMalformedResponse:
    def test_raises_on_missing_id(self, spotify_client):
        spotify_client._request = MagicMock(return_value={
            "external_urls": {"spotify": "https://open.spotify.com/playlist/abc"},
        })
        with pytest.raises(ValueError, match="malformed"):
            spotify_client.create_playlist("Test", "Desc", ["uri:1"])

    def test_raises_on_missing_url(self, spotify_client):
        spotify_client._request = MagicMock(return_value={"id": "pl1"})
        with pytest.raises(ValueError, match="malformed"):
            spotify_client.create_playlist("Test", "Desc", ["uri:1"])

    def test_raises_on_empty_response(self, spotify_client):
        spotify_client._request = MagicMock(return_value={})
        with pytest.raises(ValueError, match="malformed"):
            spotify_client.create_playlist("Test", "Desc", ["uri:1"])


# --- Issue #28: Unicode whitespace bypass ---

class TestUnicodeWhitespace:
    @pytest.fixture
    def headers(self):
        return {"Origin": "http://localhost"}

    def test_nbsp_only_prompt_rejected(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "\u00a0\u00a0"}, headers=headers)
        assert resp.status_code == 400

    def test_zero_width_space_prompt_rejected(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "\u200b"}, headers=headers)
        assert resp.status_code == 400

    def test_bom_only_prompt_rejected(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "\ufeff"}, headers=headers)
        assert resp.status_code == 400

    def test_mixed_unicode_whitespace_rejected(self, client, headers):
        resp = client.post("/api/generate", json={"prompt": "  \u00a0\u200b\ufeff  "}, headers=headers)
        assert resp.status_code == 400


# --- Issue #27: Retry-After cap ---

class TestRetryAfterCap:
    @patch("app.time.sleep")
    def test_caps_retry_after_at_10(self, mock_sleep, spotify_client):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "60"}

        success = MagicMock()
        success.status_code = 200
        success.content = b'{"ok": true}'
        success.json.return_value = {"ok": True}

        spotify_client.session.request = MagicMock(side_effect=[rate_limited, success])
        result = spotify_client._request("GET", "search")
        assert result == {"ok": True}
        mock_sleep.assert_called_once_with(10)

    @patch("app.time.sleep")
    def test_does_not_cap_small_retry_after(self, mock_sleep, spotify_client):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "3"}

        success = MagicMock()
        success.status_code = 200
        success.content = b'{"ok": true}'
        success.json.return_value = {"ok": True}

        spotify_client.session.request = MagicMock(side_effect=[rate_limited, success])
        spotify_client._request("GET", "search")
        mock_sleep.assert_called_once_with(3)


# --- Issue #29: Pluralization ---

class TestPluralisation:
    def test_count_1_uses_singular(self):
        """curate_playlist prompt should say '1 track' not '1 tracks'."""
        from app import curate_playlist
        with patch("app._anthropic_client") as mock_client:
            mock_response = MagicMock()
            mock_response.parsed_output = CuratedPlaylist(
                playlist_name="Solo", description="One", selected_uris=["uri:1"]
            )
            mock_client.messages.parse.return_value = mock_response
            curate_playlist("test", [CandidateTrack(title="S", artist="A", uri="uri:1")], count=1)
            call_args = mock_client.messages.parse.call_args
            content = call_args[1]["messages"][0]["content"]
            assert "1 track" in content
            assert "1 tracks" not in content


# --- Issue #22: Single-year era ---

class TestSingleYearEra:
    def test_build_prompt_single_year(self):
        result = build_prompt("test", era_from=2000, era_to=2000)
        assert "the year 2000" in result
        assert "2000-2000" not in result

    def test_build_prompt_range(self):
        result = build_prompt("test", era_from=1990, era_to=1999)
        assert "1990-1999" in result

    def test_discover_single_year_filter(self, spotify_client):
        spotify_client.search_tracks = MagicMock(return_value=[])
        discover_candidates(spotify_client, ["q1"], max_workers=1, era_from=2000, era_to=2000)
        # Should use "year:2000" not "year:2000-2000"
        calls = spotify_client.search_tracks.call_args_list
        filtered_query = calls[0][0][0]
        assert "year:2000" in filtered_query
        assert "year:2000-2000" not in filtered_query


# --- Request deadline ---

class TestRequestDeadline:
    def test_check_deadline_passes_when_within_limit(self):
        _check_deadline(time.monotonic())  # should not raise

    def test_check_deadline_raises_when_exceeded(self):
        # Simulate a start time far in the past
        with pytest.raises(TimeoutError):
            _check_deadline(time.monotonic() - REQUEST_DEADLINE - 1)

    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    def test_timeout_during_search_returns_504(self, mock_discover, mock_queries, mock_spotify, client):
        mock_queries.return_value = ["q1"]
        mock_discover.side_effect = TimeoutError("deadline exceeded")
        mock_spotify.return_value = MagicMock()

        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 504
        assert "too long" in resp.get_json()["error"]

    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    @patch("app.curate_playlist")
    def test_timeout_during_curation_returns_504(self, mock_curate, mock_discover, mock_queries, mock_spotify, client):
        mock_queries.return_value = ["q1"]
        mock_discover.return_value = [
            CandidateTrack(title="Song", artist="Artist", uri="spotify:track:1"),
        ]
        mock_curate.side_effect = TimeoutError("deadline exceeded")
        mock_spotify.return_value = MagicMock()

        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 504
        assert "too long" in resp.get_json()["error"]


# --- API key authentication ---

class TestAPIKeyAuth:
    @patch("app.API_KEY", "test-secret-key")
    @patch("app.get_spotify")
    @patch("app.generate_search_queries")
    @patch("app.discover_candidates")
    def test_valid_api_key_bypasses_csrf(self, mock_discover, mock_queries, mock_spotify, client):
        """A valid X-API-Key header should bypass CSRF (no Origin needed)."""
        mock_queries.return_value = ["q1"]
        mock_discover.return_value = []
        mock_spotify.return_value = MagicMock()

        resp = client.post(
            "/api/generate",
            json={"prompt": "chill vibes"},
            headers={"X-API-Key": "test-secret-key"},
        )
        # Should get past CSRF — 404 means "no tracks found" (not 403)
        assert resp.status_code == 404
        assert "No tracks found" in resp.get_json()["error"]

    @patch("app.API_KEY", "test-secret-key")
    def test_invalid_api_key_returns_401(self, client):
        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid API key" in resp.get_json()["error"]

    def test_missing_api_key_falls_through_to_csrf(self, client):
        """Without X-API-Key, the existing CSRF check applies."""
        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
        )
        assert resp.status_code == 403
        assert "Invalid request origin" in resp.get_json()["error"]

    @patch("app.API_KEY", None)
    def test_api_key_header_with_no_env_var_returns_401(self, client):
        """If API_KEY env var is not set, any X-API-Key header is rejected."""
        resp = client.post(
            "/api/generate",
            json={"prompt": "test"},
            headers={"X-API-Key": "some-key"},
        )
        assert resp.status_code == 401
