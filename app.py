"""Flask backend for the Spotify Playlist Maker web app."""

from __future__ import annotations

import hmac
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from collections import defaultdict

import anthropic
import pydantic
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, send_from_directory
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

app = Flask(__name__, static_folder="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-public playlist-modify-private"
HTTP_TIMEOUT = (5, 30)  # (connect, read) seconds for Spotify API calls
REQUEST_DEADLINE = 200  # seconds — overall limit per /api/generate request (buffer before 240s gunicorn timeout)
API_KEY = os.environ.get("API_KEY")  # Optional API key for programmatic access (Siri Shortcuts, etc.)

# Shared Anthropic client — reused across requests (thread-safe, keeps connection pool)
_anthropic_client = anthropic.Anthropic()

# Simple in-memory rate limiter: max 5 requests per IP per 60 seconds
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW = 60
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = threading.Lock()


def _check_deadline(start: float) -> None:
    """Raise TimeoutError if the request deadline has been exceeded."""
    elapsed = time.monotonic() - start
    if elapsed >= REQUEST_DEADLINE:
        raise TimeoutError(f"Request deadline exceeded ({elapsed:.0f}s >= {REQUEST_DEADLINE}s)")


def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store[ip]
        # Prune old entries
        _rate_limit_store[ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
            return True
        _rate_limit_store[ip].append(now)
        return False


# --- Models ---

class SearchQueries(pydantic.BaseModel):
    queries: list[str]


class CandidateTrack(pydantic.BaseModel):
    title: str
    artist: str
    uri: str


class CuratedPlaylist(pydantic.BaseModel):
    playlist_name: str
    description: str
    selected_uris: list[str]


# --- Spotify Client ---

class SpotifyClient:
    """Lightweight Spotify API client using raw requests with auto-refreshing tokens."""

    def __init__(self, cache_path: str) -> None:
        self.auth = SpotifyOAuth(scope=SCOPES, cache_path=cache_path)
        self.session = requests.Session()
        self._token_lock = threading.Lock()
        # Acquire token eagerly so the interactive browser auth happens once,
        # before any parallel threads try to use it.
        self.auth.get_access_token(as_dict=False)

    def _headers(self) -> dict[str, str]:
        with self._token_lock:
            token = self.auth.get_access_token(as_dict=False)
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, max_retries: int = 3, **kwargs) -> dict:
        kwargs.setdefault("timeout", HTTP_TIMEOUT)
        last_resp = None
        for attempt in range(max_retries):
            resp = self.session.request(
                method, f"{SPOTIFY_API}/{path}", headers=self._headers(), **kwargs
            )
            last_resp = resp
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                if retry_after > 10:
                    logger.warning(f"Rate limited for {retry_after}s, capping sleep at 10s")
                    retry_after = 10
                else:
                    logger.warning(f"Rate limited, retrying in {retry_after}s...")
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        # All retries exhausted — raise the last response's error
        last_resp.raise_for_status()
        return {}  # unreachable, but satisfies type checker

    def search_tracks(self, query: str, limit: int = 10) -> list[CandidateTrack]:
        data = self._request(
            "GET", "search", params={"q": query, "type": "track", "limit": limit}
        )
        candidates = []
        for item in data.get("tracks", {}).get("items", []):
            name = item.get("name")
            uri = item.get("uri")
            if not name or not uri:
                continue
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            candidates.append(CandidateTrack(
                title=name, artist=artists, uri=uri
            ))
        return candidates

    def create_playlist(self, name: str, description: str, track_uris: list[str]) -> tuple[str, int]:
        """Create a playlist and add tracks. Returns (url, tracks_added)."""
        playlist = self._request(
            "POST",
            "me/playlists",
            json={"name": name, "public": False, "description": description},
        )
        playlist_id = playlist.get("id")
        playlist_url = playlist.get("external_urls", {}).get("spotify")
        if not playlist_id or not playlist_url:
            raise ValueError("Spotify returned a malformed playlist response (missing id or URL).")
        tracks_added = 0
        for i in range(0, len(track_uris), 100):
            batch = track_uris[i : i + 100]
            try:
                self._request(
                    "POST",
                    f"playlists/{playlist_id}/items",
                    json={"uris": batch},
                )
                tracks_added += len(batch)
            except requests.HTTPError:
                logger.warning(f"Failed to add batch starting at index {i}, skipping")
        return playlist_url, tracks_added


# --- Allowed values for mood/context ---

ALLOWED_MOODS = {
    "happy", "melancholy", "nostalgic", "aggressive",
    "dreamy", "romantic", "energetic", "peaceful",
}
ALLOWED_CONTEXTS = {
    "working out", "cooking dinner", "long drive", "house party",
    "falling asleep", "morning coffee", "studying", "date night",
}
ENERGY_LABELS = {1: "very low energy/calm", 2: "low energy/relaxed", 3: "moderate energy", 4: "high energy/upbeat", 5: "very high energy/intense"}


def build_prompt(
    prompt: str,
    energy: Optional[int] = None,
    moods: Optional[list[str]] = None,
    context: Optional[str] = None,
    era_from: Optional[int] = None,
    era_to: Optional[int] = None,
    seed: Optional[str] = None,
) -> str:
    """Build an enriched prompt string from the base prompt and optional mood/context fields."""
    parts = [prompt]
    if energy is not None:
        parts.append(f"Energy level: {ENERGY_LABELS.get(energy, 'moderate energy')}.")
    if moods:
        parts.append(f"Mood: {', '.join(moods)}.")
    if context:
        parts.append(f"Context/setting: {context}.")
    if era_from is not None and era_to is not None:
        if era_from == era_to:
            parts.append(f"Era: only tracks from the year {era_from}.")
        else:
            parts.append(f"Era: only tracks from {era_from}-{era_to}.")
    if seed:
        parts.append(f"Use '{seed}' as a style/sound reference.")
    return " ".join(parts)


# --- AI Functions ---

def generate_search_queries(prompt: str, count: int = 20, max_attempts: int = 2) -> list[str]:
    # Scale query count for larger playlists to build a bigger candidate pool
    if count <= 20:
        query_range = "8-12"
    elif count <= 35:
        query_range = "12-16"
    else:
        query_range = "16-20"

    for attempt in range(max_attempts):
        try:
            response = _anthropic_client.messages.parse(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"I want to build a Spotify playlist for: \"{prompt}\"\n\n"
                            f"Generate {query_range} diverse Spotify search queries that would find tracks "
                            "matching this vibe. Include variations like:\n"
                            "- Direct searches (e.g., \"final fantasy VII lofi\")\n"
                            "- Artist-specific searches for artists known in this space\n"
                            "- Genre + theme combinations\n"
                            "- Related keywords and synonyms\n\n"
                            "If the prompt specifies an era or time period, make sure your queries "
                            "reference artists, genres, and styles from that era.\n\n"
                            "Each query should be a real Spotify search string."
                        ),
                    }
                ],
                output_format=SearchQueries,
            )
            return response.parsed_output.queries
        except pydantic.ValidationError:
            if attempt < max_attempts - 1:
                logger.warning("Query generation returned unparseable output, retrying...")
                continue
            raise ValueError("Failed to generate search queries. Please try again.")


def discover_candidates(
    spotify: SpotifyClient,
    queries: list[str],
    max_workers: int = 5,
    era_from: Optional[int] = None,
    era_to: Optional[int] = None,
    deadline_start: Optional[float] = None,
) -> list[CandidateTrack]:
    seen_uris: set[str] = set()
    candidates: list[CandidateTrack] = []

    # Run queries both WITH and WITHOUT year filter when era is set.
    # The year-filtered queries find era-specific hits; the unfiltered queries
    # provide fallback candidates that curation can still filter by era.
    if era_from is not None and era_to is not None:
        year_filter = f" year:{era_from}" if era_from == era_to else f" year:{era_from}-{era_to}"
        filtered_queries = [q + year_filter for q in queries]
        queries = filtered_queries + queries

    def _search(query: str) -> list[CandidateTrack]:
        try:
            if deadline_start is not None:
                _check_deadline(deadline_start)
            return spotify.search_tracks(query, limit=10)
        except (requests.HTTPError, requests.Timeout) as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for results in pool.map(_search, queries):
            for track in results:
                if track.uri not in seen_uris:
                    seen_uris.add(track.uri)
                    candidates.append(track)

    if deadline_start is not None:
        _check_deadline(deadline_start)

    return candidates


def curate_playlist(
    prompt: str, candidates: list[CandidateTrack], count: int, max_attempts: int = 2,
    deadline_start: Optional[float] = None,
) -> CuratedPlaylist:
    track_list = "\n".join(
        f"- [{t.uri}] {t.title} — {t.artist}" for t in candidates
    )

    # When the candidate pool is small relative to the requested count,
    # soften the curation to avoid returning only 2-3 tracks.
    if len(candidates) < count * 2:
        inclusivity_note = (
            "\n\nIMPORTANT: The candidate pool is small relative to the requested "
            f"count ({len(candidates)} candidates for {count} slots). Be more "
            "inclusive — select tracks that are a reasonable fit, not just perfect "
            "matches. It's better to fill the playlist than to be overly selective."
        )
    else:
        inclusivity_note = ""

    for attempt in range(max_attempts):
        if deadline_start is not None:
            _check_deadline(deadline_start)
        try:
            response = _anthropic_client.messages.parse(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"I'm building a Spotify playlist for: \"{prompt}\"\n\n"
                            f"Here are {len(candidates)} candidate tracks found on Spotify:\n"
                            f"{track_list}\n\n"
                            f"Select exactly {count} {'track' if count == 1 else 'tracks'} (or fewer if not enough good matches) "
                            "that best fit the requested vibe. Rules:\n"
                            "- ONLY select tracks that genuinely match the prompt's theme/genre/mood\n"
                            "- Skip any track that feels off-theme, even if it's a good song\n"
                            "- Prefer variety in artists when possible\n"
                            "- Return the spotify URIs of your selections in selected_uris\n\n"
                            "Also create a playlist name and description matching the mood."
                            f"{inclusivity_note}"
                        ),
                    }
                ],
                output_format=CuratedPlaylist,
            )
            return response.parsed_output
        except pydantic.ValidationError:
            if attempt < max_attempts - 1:
                logger.warning("Curation returned unparseable output, retrying...")
                continue
            raise ValueError("Failed to curate playlist. Please try again.")


# --- Initialize Spotify client at startup ---

_spotify: Optional[SpotifyClient] = None
_spotify_lock = threading.Lock()


def get_spotify() -> SpotifyClient:
    global _spotify
    if _spotify is None:
        with _spotify_lock:
            if _spotify is None:
                # On Azure, the app runs from /tmp but persistent files are in /home/site/wwwroot
                home_wwwroot = "/home/site/wwwroot"
                if os.path.isdir(home_wwwroot):
                    cache_path = os.path.join(home_wwwroot, ".spotify_cache")
                else:
                    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spotify_cache")
                _spotify = SpotifyClient(cache_path)
                logger.info("Spotify client initialized.")
    return _spotify


# --- Routes ---

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


TRUSTED_IMAGE_HOSTS = {"blob.core.windows.net", "windows.net"}

@app.route("/api/daily-image")
def daily_image():
    url = os.environ.get("DAILY_IMAGE_URL")
    if not url:
        return "", 204
    parsed = urlparse(url)
    if not any(parsed.hostname and parsed.hostname.endswith(h) for h in TRUSTED_IMAGE_HOSTS):
        logger.warning(f"Blocked redirect to untrusted host: {parsed.hostname}")
        return "", 204
    return redirect(url)


@app.route("/api/generate", methods=["POST"])
def generate():
    # API key authentication — allows programmatic callers (Siri Shortcuts, etc.)
    # to bypass CSRF. If the header is present it must be valid; if absent, fall
    # through to the normal Origin/Referer CSRF check for browser requests.
    api_key = request.headers.get("X-API-Key")
    if api_key:
        if not API_KEY or not hmac.compare_digest(api_key, API_KEY):
            return jsonify({"error": "Invalid API key."}), 401
    else:
        # CSRF protection: verify Origin or Referer hostname matches our host.
        # Compare hostnames only — behind Azure's reverse proxy, the scheme and
        # port in request.host_url may differ from the browser's Origin header.
        origin = request.headers.get("Origin") or ""
        referer = request.headers.get("Referer") or ""
        expected_host = request.host.split(":")[0]  # hostname without port
        if origin:
            origin_host = urlparse(origin).hostname or ""
            if origin_host != expected_host:
                return jsonify({"error": "Invalid request origin."}), 403
        elif referer:
            referer_host = urlparse(referer).hostname or ""
            if referer_host != expected_host:
                return jsonify({"error": "Invalid request origin."}), 403
        else:
            return jsonify({"error": "Invalid request origin."}), 403

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if _is_rate_limited(client_ip):
        return jsonify({"error": "Too many requests. Please wait a minute."}), 429

    data = request.get_json()
    if not data or not data.get("prompt"):
        return jsonify({"error": "Missing 'prompt' field"}), 400

    prompt = re.sub(r'[\s\u200b\u00a0\ufeff]+', ' ', str(data["prompt"])).strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400
    if len(prompt) > 500:
        return jsonify({"error": "Prompt is too long (max 500 characters)."}), 400

    try:
        count = max(1, min(50, int(data.get("count", 20))))
    except (TypeError, ValueError):
        count = 20

    # Parse optional mood/context fields
    energy = None
    raw_energy = data.get("energy")
    if raw_energy is not None:
        try:
            energy = int(raw_energy)
            if energy < 1 or energy > 5:
                return jsonify({"error": "Energy must be between 1 and 5."}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "Energy must be an integer."}), 400

    moods = None
    raw_moods = data.get("moods")
    if raw_moods:
        if not isinstance(raw_moods, list) or len(raw_moods) > 3:
            return jsonify({"error": "Moods must be a list of up to 3 tags."}), 400
        moods = [str(m).lower().strip() for m in raw_moods]
        if any(m not in ALLOWED_MOODS for m in moods):
            return jsonify({"error": f"Invalid mood. Allowed: {', '.join(sorted(ALLOWED_MOODS))}"}), 400

    context = None
    raw_context = data.get("context")
    if raw_context:
        context = str(raw_context).lower().strip()
        if context not in ALLOWED_CONTEXTS:
            return jsonify({"error": f"Invalid context. Allowed: {', '.join(sorted(ALLOWED_CONTEXTS))}"}), 400

    era_from = None
    era_to = None
    raw_era_from = data.get("era_from")
    raw_era_to = data.get("era_to")
    if raw_era_from is not None and raw_era_to is not None:
        try:
            era_from = int(raw_era_from)
            era_to = int(raw_era_to)
            if era_from < 1900 or era_to > datetime.now().year or era_from > era_to:
                return jsonify({"error": "Invalid era range."}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "Era years must be integers."}), 400

    seed = None
    raw_seed = data.get("seed")
    if raw_seed:
        seed = str(raw_seed).strip()[:100] or None

    # Build enriched prompt
    enriched_prompt = build_prompt(prompt, energy, moods, context, era_from, era_to, seed)

    try:
        spotify = get_spotify()
    except Exception:
        logger.exception("Spotify auth failed")
        return jsonify({"error": "Spotify authentication failed. Please try again later."}), 500

    deadline_start = time.monotonic()

    try:
        # Step 1: Generate search queries (scaled by count for larger playlists)
        queries = generate_search_queries(enriched_prompt, count=count)
        logger.info(f"Generated {len(queries)} search queries for: {prompt}")

        # Step 2: Search Spotify
        candidates = discover_candidates(spotify, queries, era_from=era_from, era_to=era_to, deadline_start=deadline_start)
        logger.info(f"Found {len(candidates)} unique candidate tracks")

        if not candidates:
            return jsonify({"error": "No tracks found on Spotify. Try a broader genre or more well-known artists."}), 404

        # Step 3: Curate
        curated = curate_playlist(enriched_prompt, candidates, count, deadline_start=deadline_start)

        # Filter to valid URIs
        valid_uris = {c.uri for c in candidates}
        track_uris = [uri for uri in curated.selected_uris if uri in valid_uris]

        if not track_uris:
            return jsonify({"error": "No matching tracks passed curation. Try a broader prompt or different keywords."}), 404

        # Step 4: Create playlist
        uri_to_track = {c.uri: c for c in candidates}
        url, tracks_added = spotify.create_playlist(curated.playlist_name, curated.description, track_uris)
        logger.info(f"Created playlist '{curated.playlist_name}': {tracks_added}/{len(track_uris)} tracks added")

        tracks = [
            {"title": uri_to_track[uri].title, "artist": uri_to_track[uri].artist}
            for uri in track_uris if uri in uri_to_track
        ]

        response = {
            "playlist_url": url,
            "playlist_name": curated.playlist_name,
            "description": curated.description,
            "tracks": tracks,
            "tracks_found": len(track_uris),
        }
        if len(track_uris) < count:
            response["warning"] = (
                f"Only {len(track_uris)} {'track' if len(track_uris) == 1 else 'tracks'} "
                f"matched your criteria (you requested {count})."
            )
        return jsonify(response)

    except TimeoutError:
        logger.warning(f"Request deadline exceeded for prompt: {prompt}")
        return jsonify({"error": "Request took too long. Try a simpler prompt or fewer tracks."}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid Anthropic API key."}), 500
    except anthropic.APIConnectionError:
        return jsonify({"error": "Could not connect to Anthropic API."}), 500
    except requests.HTTPError:
        logger.exception("Spotify API error")
        return jsonify({"error": "Spotify API error. Please try again."}), 500
    except Exception:
        logger.exception("Unexpected error")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# Pre-initialize Spotify client on startup (works under both gunicorn and dev server)
try:
    get_spotify()
except Exception:
    logger.warning("Spotify client init deferred — will retry on first request.")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
