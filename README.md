# Playlist Maker

An AI-powered Spotify playlist generator. Describe a vibe in plain English and it creates a real playlist on your Spotify account.

**How it works:**
1. Claude generates diverse Spotify search queries from your prompt
2. Searches are run in parallel across the Spotify catalog
3. Claude curates the best-matching tracks from the results
4. A playlist is created on your Spotify account

Built with Flask, Claude (Anthropic API), and the Spotify Web API. Deployed on Azure App Service with GitHub Actions CI/CD.

## Tech Stack

- **Backend:** Python / Flask / Gunicorn
- **AI:** Claude Sonnet (query generation) + Claude Opus (curation)
- **Music:** Spotify Web API via `spotipy` OAuth
- **Infra:** Azure App Service (Linux), GitHub Actions
- **Frontend:** Vanilla HTML/CSS/JS — no build step

## Running Locally

```bash
git clone https://github.com/rkemery/playlist-maker.git
cd playlist-maker
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```

You'll need:
- An [Anthropic API key](https://console.anthropic.com/)
- A [Spotify Developer app](https://developer.spotify.com/dashboard) with the redirect URI set
- Spotify Premium (required for dev mode apps)

```bash
python app.py
# Open http://localhost:5000
```

On first run, Spotify OAuth will open a browser window to authorize. The token is cached in `.spotify_cache` and auto-refreshes.

## Project Structure

```
playlist_maker/
  app.py              # Flask backend + Spotify/Claude integration
  startup.sh          # Gunicorn entrypoint for Azure
  requirements.txt
  static/
    index.html        # Frontend (single file, no build)
  .github/
    workflows/
      deploy.yml      # CI/CD — pushes to main auto-deploy
```

## Architecture

```
Browser  -->  Flask (Azure App Service)  -->  Spotify API
                    |                          search + create playlist
                    v
              Anthropic API
              generate queries + curate tracks
```

The CLI version (`playlist_maker.py` in my home directory) does the same thing from the terminal — this repo is the web wrapper.

## Deployment

Pushes to `main` auto-deploy to Azure App Service via GitHub Actions. The workflow uses Azure publish profile credentials stored as a GitHub secret.

Spotify auth requires a one-time interactive OAuth flow. The resulting `.spotify_cache` token file is uploaded to Azure's persistent storage (`/home/site/wwwroot/`) and auto-refreshes indefinitely.

## License

MIT
