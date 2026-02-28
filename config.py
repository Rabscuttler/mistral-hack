"""Configuration for the song lyrics pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

HF_USERNAME = os.environ.get("HF_USERNAME", "laurence-furbnow")

BASELINE_MODEL = "mistral-medium-latest"
FINETUNED_MODEL = os.environ.get("HF_MODEL_REPO", f"{HF_USERNAME}/mistral-7b-lyrics-lora")
WEAVE_PROJECT = "mistral-hackathon"

GENRES = ["pop", "country", "rock", "indie"]

THEMES: dict[str, list[str]] = {
    "pop": [
        "falling in love at first sight",
        "dancing alone in your bedroom",
        "summer road trip with friends",
        "getting over a breakup",
        "chasing your dreams in a big city",
        "a party that changes everything",
        "missing someone far away",
        "feeling unstoppable and confident",
        "a secret romance",
        "growing up and leaving home",
    ],
    "country": [
        "a small town Friday night",
        "driving down a dirt road at sunset",
        "falling for your best friend",
        "a soldier coming home to family",
        "losing someone you love too soon",
        "the simple life and front porch memories",
        "a honky-tonk heartbreak",
        "faith and redemption after hard times",
        "raising kids in the countryside",
        "a love letter to your hometown",
    ],
    "rock": [
        "rebellion against the system",
        "driving through the desert at midnight",
        "a love that burns too hot",
        "the end of the world as we know it",
        "fighting your inner demons",
        "nostalgia for wilder days",
        "a soldier coming home",
        "freedom on the open road",
        "losing your mind in isolation",
        "standing your ground no matter what",
    ],
    "indie": [
        "a rainy afternoon in a coffee shop",
        "the quiet beauty of ordinary life",
        "unrequited love from a distance",
        "walking through a city at 3am",
        "seasons changing as a metaphor for growth",
        "a friendship slowly falling apart",
        "finding peace in solitude",
        "the bittersweet feeling of nostalgia",
        "a conversation you never had",
        "small moments that mean everything",
    ],
}

# HuggingFace configuration
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", f"{HF_USERNAME}/lyrics-sft")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", f"{HF_USERNAME}/mistral-7b-lyrics-lora")
DATASET_ZIP_PATH = "~/Downloads/genius-dataset.zip"

# Data filtering
MIN_LYRICS_LINES = 4
MAX_LYRICS_LINES = 200
# Maps our genre names -> CSV tag values in the dataset
GENRE_MAP = {
    "pop": "pop",
    "country": "country",
    "rock": "rock",
    "indie": "rock",  # No indie tag in dataset; use rock as closest match
}
