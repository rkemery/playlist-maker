"""Flask backend for the Spotify Playlist Maker web app."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import anthropic
import pydantic
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

app = Flask(__name__, static_folder="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-public playlist-modify-private"


# --- Models ---

class Song(pydantic.BaseModel):
    title: str
    artist: str


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
        # Acquire token eagerly so the interactive browser auth happens once,
        # before any parallel threads try to use it.
        self.auth.get_access_token(as_dict=False)

    def _headers(self) -> dict[str, str]:
        token = self.auth.get_access_token(as_dict=False)
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, max_retries: int = 3, **kwargs) -> dict:
        for attempt in range(max_retries):
            resp = self.session.request(
                method, f"{SPOTIFY_API}/{path}", headers=self._headers(), **kwargs
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                logger.warning(f"Rate limited, retrying in {retry_after}s...")
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        resp.raise_for_status()
        return {}

    def search_tracks(self, query: str, limit: int = 10) -> list[CandidateTrack]:
        data = self._request(
            "GET", "search", params={"q": query, "type": "track", "limit": limit}
        )
        candidates = []
        for item in data.get("tracks", {}).get("items", []):
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            candidates.append(CandidateTrack(
                title=item["name"], artist=artists, uri=item["uri"]
            ))
        return candidates

    def create_playlist(self, name: str, description: str, track_uris: list[str]) -> str:
        playlist = self._request(
            "POST",
            "me/playlists",
            json={"name": name, "public": False, "description": description},
        )
        for i in range(0, len(track_uris), 100):
            self._request(
                "POST",
                f"playlists/{playlist['id']}/items",
                json={"uris": track_uris[i : i + 100]},
            )
        return playlist["external_urls"]["spotify"]


# --- AI Functions ---

def generate_search_queries(prompt: str) -> list[str]:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"I want to build a Spotify playlist for: \"{prompt}\"\n\n"
                    "Generate 8-12 diverse Spotify search queries that would find tracks "
                    "matching this vibe. Include variations like:\n"
                    "- Direct searches (e.g., \"final fantasy VII lofi\")\n"
                    "- Artist-specific searches for artists known in this space\n"
                    "- Genre + theme combinations\n"
                    "- Related keywords and synonyms\n\n"
                    "Each query should be a real Spotify search string."
                ),
            }
        ],
        output_format=SearchQueries,
    )
    return response.parsed_output.queries


def discover_candidates(
    spotify: SpotifyClient, queries: list[str], max_workers: int = 5
) -> list[CandidateTrack]:
    seen_uris: set[str] = set()
    candidates: list[CandidateTrack] = []

    def _search(query: str) -> list[CandidateTrack]:
        return spotify.search_tracks(query, limit=10)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for results in pool.map(_search, queries):
            for track in results:
                if track.uri not in seen_uris:
                    seen_uris.add(track.uri)
                    candidates.append(track)
    return candidates


def curate_playlist(
    prompt: str, candidates: list[CandidateTrack], count: int
) -> CuratedPlaylist:
    client = anthropic.Anthropic()
    track_list = "\n".join(
        f"- [{t.uri}] {t.title} — {t.artist}" for t in candidates
    )
    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"I'm building a Spotify playlist for: \"{prompt}\"\n\n"
                    f"Here are {len(candidates)} candidate tracks found on Spotify:\n"
                    f"{track_list}\n\n"
                    f"Select exactly {count} tracks (or fewer if not enough good matches) "
                    "that best fit the requested vibe. Rules:\n"
                    "- ONLY select tracks that genuinely match the prompt's theme/genre/mood\n"
                    "- Skip any track that feels off-theme, even if it's a good song\n"
                    "- Prefer variety in artists when possible\n"
                    "- Return the spotify URIs of your selections in selected_uris\n\n"
                    "Also create a playlist name and description matching the mood."
                ),
            }
        ],
        output_format=CuratedPlaylist,
    )
    return response.parsed_output


# --- Initialize Spotify client at startup ---

_spotify: Optional[SpotifyClient] = None


def get_spotify() -> SpotifyClient:
    global _spotify
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


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or not data.get("prompt"):
        return jsonify({"error": "Missing 'prompt' field"}), 400

    prompt = data["prompt"]
    count = data.get("count", 20)
    count = max(1, min(50, int(count)))

    try:
        spotify = get_spotify()
    except Exception as e:
        logger.exception("Spotify auth failed")
        return jsonify({"error": f"Spotify authentication error: {e}"}), 500

    try:
        # Step 1: Generate search queries
        queries = generate_search_queries(prompt)
        logger.info(f"Generated {len(queries)} search queries for: {prompt}")

        # Step 2: Search Spotify
        candidates = discover_candidates(spotify, queries)
        logger.info(f"Found {len(candidates)} unique candidate tracks")

        if not candidates:
            return jsonify({"error": "No tracks found on Spotify. Try a different prompt."}), 404

        # Step 3: Curate
        curated = curate_playlist(prompt, candidates, count)

        # Filter to valid URIs
        valid_uris = {c.uri for c in candidates}
        track_uris = [uri for uri in curated.selected_uris if uri in valid_uris]

        if not track_uris:
            return jsonify({"error": "No matching tracks found. Try a different prompt."}), 404

        # Step 4: Create playlist
        uri_to_track = {c.uri: c for c in candidates}
        url = spotify.create_playlist(curated.playlist_name, curated.description, track_uris)

        tracks = [
            {"title": uri_to_track[uri].title, "artist": uri_to_track[uri].artist}
            for uri in track_uris if uri in uri_to_track
        ]

        return jsonify({
            "playlist_url": url,
            "playlist_name": curated.playlist_name,
            "description": curated.description,
            "tracks": tracks,
        })

    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid Anthropic API key."}), 500
    except anthropic.APIConnectionError:
        return jsonify({"error": "Could not connect to Anthropic API."}), 500
    except requests.HTTPError as e:
        logger.exception("Spotify API error")
        return jsonify({"error": f"Spotify API error: {e}"}), 500
    except Exception as e:
        logger.exception("Unexpected error")
        return jsonify({"error": f"Unexpected error: {e}"}), 500


if __name__ == "__main__":
    # Pre-initialize Spotify client on startup
    get_spotify()
    app.run(debug=True, port=5000)
