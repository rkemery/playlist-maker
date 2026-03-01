# Playlist Maker — Claude Code Instructions

## What This Is

AI-powered Spotify playlist generator. Users describe a vibe in plain English (web UI or API), Claude generates search queries, searches Spotify in parallel, Claude curates the best tracks, and a real playlist is created on the user's Spotify account.

## Tech Stack

- **Backend:** Python / Flask / Gunicorn (single file: `app.py`)
- **AI:** Claude Sonnet (query generation) → Claude Opus (curation)
- **Music:** Spotify Web API via `spotipy` OAuth
- **Infra:** Azure App Service (Linux), GitHub Actions CI/CD
- **Frontend:** Vanilla HTML/CSS/JS in `static/index.html` — no build step

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Entire backend — routes, AI calls, Spotify integration, context-aware helpers |
| `test_app.py` | Comprehensive pytest suite |
| `static/index.html` | Web UI (single file) |
| `startup.sh` | Gunicorn entrypoint for Azure |
| `requirements.txt` | Python dependencies |
| `.github/workflows/deploy.yml` | CI/CD: push to `main` → deploy to prod |
| `.github/workflows/deploy-dev.yml` | CI/CD: push to `dev` → deploy to dev |

## Environments

| Env | URL | Azure App | Resource Group |
|---|---|---|---|
| **Prod** | `rkemery-playlist-maker.azurewebsites.net` | `rkemery-playlist-maker` | `rg-rlkemery-5175` |
| **Dev** | `rkemery-playlist-maker-dev.azurewebsites.net` | `rkemery-playlist-maker-dev` | `rg-rlkemery-5175` |

## Git Workflow

- `main` — production. Pushes auto-deploy via `deploy.yml`
- `dev` — staging. Pushes auto-deploy via `deploy-dev.yml`
- Work on feature branches or worktrees, push to `dev` for testing, merge to `main` for release
- Use `gh release create` for tagged releases (current: v1.1.0)

## API Endpoints

### `POST /generate` (web UI form submission)
- Standard form-based playlist creation from the web UI
- Returns redirect to Spotify playlist

### `POST /api/generate` (JSON API)
- Protected by `X-API-Key` header (env var `API_KEY`)
- Used by Siri Shortcuts and other programmatic clients

**Request body:**
```json
{
  "prompt": "chill lofi vibes",
  "count": 25,
  "energy": 3,
  "moods": ["chill", "mellow"],
  "context": "for studying",
  "era_from": 2010,
  "era_to": 2024,
  "seed": "Norah Jones",
  "context_aware": true,
  "timezone": "America/Indiana/Indianapolis",
  "location": "Indianapolis, IN",
  "weather": "72F, partly cloudy"
}
```

**Response includes:** `playlist_url`, `spotify_uri`, `playlist_name`, `description`, `tracks`, `tracks_found`, `context`

### Context-Aware Mode
When `context_aware: true`, the server gathers temporal signals (time of day, day of week, season, nearby holidays) and combines with optional client-provided `location`, `weather`, `timezone`. These signals are woven into the AI prompt and playlist naming.

Key helpers in `app.py`:
- `gather_context_signals()` — builds context string from temporal + client signals
- `_get_nearby_holiday()` — checks fixed holidays + Thanksgiving with 3-day lookahead
- `_get_season()` — approximate equinox/solstice dates
- `_get_time_of_day()` — morning/afternoon/evening/late night classification

## Architecture Flow

```
Client → Flask → Claude Sonnet (generate search queries)
                → Spotify API (parallel search)
                → Claude Opus (curate tracks, name playlist)
                → Spotify API (create playlist, add tracks)
                → Response with playlist URL/URI
```

## Request Deadline

All requests have a server-side time budget (`REQUEST_DEADLINE_SECONDS`, default 110s). AI calls and Spotify operations check remaining time and abort gracefully if the deadline is near, preventing Gunicorn worker timeouts.

## Running Tests

```bash
python3 -m pytest test_app.py -v
```

Tests mock all external services (Anthropic, Spotify). No API keys needed to run tests.

## Running Locally

```bash
pip install -r requirements.txt
# Create .env with ANTHROPIC_API_KEY, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
python app.py
# Open http://localhost:5000
```

## Common Operations

```bash
# Check deployment status
gh run list --workflow=deploy-dev.yml --limit=3
gh run list --workflow=deploy.yml --limit=3

# View Azure logs
az webapp log download --name rkemery-playlist-maker-dev --resource-group rg-rlkemery-5175 --log-file /tmp/dev-logs.zip

# Quick health check
curl -s -o /dev/null -w "%{http_code}" https://rkemery-playlist-maker-dev.azurewebsites.net/
curl -s -o /dev/null -w "%{http_code}" https://rkemery-playlist-maker.azurewebsites.net/
```

## Important Notes

- `app.py` is a single-file backend — all logic lives there
- No external dependencies for context-aware features (stdlib only: `calendar`, `zoneinfo`, `datetime`)
- Spotify OAuth token is cached in `.spotify_cache` on Azure persistent storage — auto-refreshes
- The web UI and API are independent — UI changes don't affect API and vice versa
- Dev API key is set as an Azure app setting (`API_KEY`)
