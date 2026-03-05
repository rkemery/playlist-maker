# Playlist Maker — Claude Code Instructions

## What This Is

AI-powered Spotify playlist generator. Users describe a vibe in plain English (web UI or API), Claude generates search queries, searches Spotify in parallel, Claude curates the best tracks, and a real playlist is created on the user's Spotify account.

## Tech Stack

- **Backend:** Python / Flask / Gunicorn (single file: `app.py`)
- **AI:** Claude Sonnet (query generation + curation)
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
  "weather": "72F, partly cloudy",
  "use_library": true
}
```

**Response includes:** `playlist_url`, `spotify_uri`, `playlist_name`, `description`, `tracks`, `tracks_found`, `context`

### Library Mode
When `use_library: true`, the server fetches a random sample of 50 liked/saved songs from the user's Spotify library and includes them as taste context in the AI prompts. This helps Claude generate search queries and curate tracks that match the user's actual listening preferences.

Key helpers in `app.py`:
- `SpotifyClient.get_liked_tracks()` — fetches random sample from `GET /me/tracks`
- `format_library_context()` — formats liked tracks into a compact taste-profile string

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
                → Claude Sonnet (curate tracks, name playlist)
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

## Spotify Token Management

The app uses OAuth tokens cached in `.spotify_cache` on Azure persistent storage (`/home/site/wwwroot/.spotify_cache`). Tokens auto-refresh, but re-authentication is needed when OAuth scopes change (e.g., adding `user-library-read`).

### When to Re-auth

- After adding/removing scopes in `SCOPES` (line 34 of `app.py`)
- If the cached token becomes invalid or corrupted

### Re-auth Process

Since Azure doesn't have a browser for interactive OAuth, generate the token locally and upload:

1. **Get Spotify credentials from Azure** (stored as app settings, not in code):
   ```bash
   az webapp config appsettings list --name rkemery-playlist-maker-dev \
     --resource-group rg-rlkemery-5175 --query "[?name=='SPOTIPY_CLIENT_ID' || name=='SPOTIPY_CLIENT_SECRET' || name=='SPOTIPY_REDIRECT_URI'].{name:name, value:value}"
   ```

2. **Set credentials locally** (export as env vars — do not commit):
   ```bash
   export SPOTIPY_CLIENT_ID="<from step 1>"
   export SPOTIPY_CLIENT_SECRET="<from step 1>"
   export SPOTIPY_REDIRECT_URI="<from step 1>"
   ```

3. **Generate auth URL** using Python:
   ```python
   from spotipy.oauth2 import SpotifyOAuth
   auth = SpotifyOAuth(scope="playlist-modify-public playlist-modify-private user-library-read")
   print(auth.get_authorize_url())
   ```

4. **Open the URL in a browser**, authorize the app, then copy the full callback URL from the browser address bar (it will contain a `code` parameter).

5. **Exchange the code for a token**:
   ```python
   import re, json
   callback_url = "<paste the callback URL here>"
   code = re.search(r'code=([^&]+)', callback_url).group(1)
   token_info = auth.get_access_token(code)
   with open(".spotify_cache", "w") as f:
       json.dump(token_info, f)
   ```

6. **Upload to Azure** (both dev and prod):
   ```bash
   # Dev
   curl -X PUT "https://rkemery-playlist-maker-dev.scm.azurewebsites.net/api/vfs/site/wwwroot/.spotify_cache" \
     -u '<deployment-credentials>' \
     -H "If-Match: *" -H "Content-Type: application/json" \
     --data-binary @.spotify_cache

   # Prod
   curl -X PUT "https://rkemery-playlist-maker.scm.azurewebsites.net/api/vfs/site/wwwroot/.spotify_cache" \
     -u '<deployment-credentials>' \
     -H "If-Match: *" -H "Content-Type: application/json" \
     --data-binary @.spotify_cache
   ```

   **Tip:** Kudu deployment credentials can be found in the Azure Portal under the app's Deployment Center, or use `az webapp deployment list-publishing-credentials`.

7. **Clean up** — delete the local `.spotify_cache` and unset env vars:
   ```bash
   rm .spotify_cache
   unset SPOTIPY_CLIENT_ID SPOTIPY_CLIENT_SECRET SPOTIPY_REDIRECT_URI
   ```

### Redirect URI

The `SPOTIPY_REDIRECT_URI` app setting on Azure must match exactly what's configured in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard). If you get `INVALID_CLIENT: Invalid redirect URI`, update the dashboard to match.

## Important Notes

- `app.py` is a single-file backend — all logic lives there
- No external dependencies for context-aware features (stdlib only: `calendar`, `zoneinfo`, `datetime`)
- Spotify OAuth token is cached in `.spotify_cache` on Azure persistent storage — auto-refreshes
- The web UI and API are independent — UI changes don't affect API and vice versa
- Dev API key is set as an Azure app setting (`API_KEY`)
