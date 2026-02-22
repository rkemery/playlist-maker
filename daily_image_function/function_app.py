"""Azure Function: Generate a daily retro background image using OpenAI."""

from __future__ import annotations

import base64
import io
import logging
import os
from datetime import datetime, timezone

import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import OpenAI
from PIL import Image

app = func.FunctionApp()
logger = logging.getLogger(__name__)

PROMPTS = [
    "Neon-lit retro arcade machines in a dark room, 80s synthwave aesthetic, pixel art style, vibrant pink and cyan neon glow",
    "VHS tape glitch art with scanlines, 90s aesthetic, purple and magenta gradients, retro TV static",
    "Retro boombox on a city sidewalk at sunset, 80s hip-hop culture, warm orange and purple sky, graffiti wall",
    "Vintage vinyl record collection close-up, warm analog tones, 70s record store vibes, soft bokeh lighting",
    "Cassette tape with unspooled ribbon forming abstract patterns, pastel vaporwave colors, dreamy and nostalgic",
    "80s sports car on an empty highway at dusk, synthwave sunset, palm tree silhouettes, chrome reflections",
    "Retro CRT television displaying colorful static, dark room, neon tube lights, cyberpunk aesthetic",
    "Vintage stereo equalizer bars glowing green, dark background, analog music equipment, warm amber accents",
    "90s skateboard park at golden hour, grunge aesthetic, concrete ramps, faded film photography style",
    "Neon sign of a music note in a rainy alley, 80s noir, wet reflections on pavement, moody blue and pink",
    "Retro roller skating rink with disco ball, 70s funk vibes, multicolored floor lights, motion blur",
    "Old school headphones on a turntable, warm vintage tones, vinyl groove texture, cozy listening session",
    "Pixel art cityscape at night, 16-bit video game style, neon building lights, starry sky above",
    "Vintage jukebox in a 50s diner, chrome and neon, checkered floor, warm nostalgic lighting",
    "Abstract geometric shapes in Memphis design style, bold 80s colors, triangles circles and squiggles",
    "Retro computer terminal with green phosphor text, dark room, floppy disks scattered, hacker aesthetic",
    "90s rave flyer aesthetic, abstract fractals, acid house smiley, UV reactive colors on black",
    "Vintage Polaroid photos scattered on a wooden table, faded colors, summer memories, soft natural light",
    "Neon palm trees against a gradient sunset sky, outrun aesthetic, grid landscape, retro futurism",
    "Old boombox surrounded by cassette tapes on a concrete floor, urban 90s, warm afternoon light, lo-fi vibes",
]

CONTAINER_NAME = "playlist-maker-images"
BLOB_NAME = "daily-bg.webp"


@app.timer_trigger(schedule="0 0 6 * * *", arg_name="timer", run_on_startup=False)
def generate_daily_image(timer: func.TimerRequest) -> None:
    """Generate a retro background image and upload to Azure Blob Storage."""
    if timer.past_due:
        logger.info("Timer is past due, running anyway.")

    # Pick prompt based on day of year
    day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
    prompt = PROMPTS[day_of_year % len(PROMPTS)]
    logger.info(f"Using prompt index {day_of_year % len(PROMPTS)}: {prompt[:60]}...")

    # Generate image via OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        n=1,
        size="1536x1024",
        quality="low",
    )

    # Decode base64 image data
    image_data = base64.b64decode(response.data[0].b64_json)
    logger.info(f"Generated image: {len(image_data)} bytes (PNG)")

    # Convert PNG to WebP via Pillow for compression
    img = Image.open(io.BytesIO(image_data))
    webp_buffer = io.BytesIO()
    img.save(webp_buffer, format="WEBP", quality=75)
    webp_bytes = webp_buffer.getvalue()
    logger.info(f"Compressed to WebP: {len(webp_bytes)} bytes")

    # Upload to Azure Blob Storage
    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(CONTAINER_NAME, BLOB_NAME)
    blob_client.upload_blob(
        webp_bytes,
        overwrite=True,
        content_settings=ContentSettings(
            content_type="image/webp",
            cache_control="public, max-age=86400",
        ),
    )
    logger.info(f"Uploaded {BLOB_NAME} to {CONTAINER_NAME}")
