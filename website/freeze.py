"""Freeze the website into static files for GitHub Pages.

Usage:  python3 website/freeze.py
Output: docs/
"""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template
from flask_frozen import Freezer

WEBSITE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEBSITE_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
PAIRS_PATH = WEBSITE_DIR / "pairs.json"

# --- Flask app for freezing ---

app = Flask(
    __name__,
    template_folder=str(WEBSITE_DIR / "templates"),
    static_folder=str(WEBSITE_DIR / "static"),
)
app.config["FREEZER_DESTINATION"] = str(PROJECT_DIR / "docs")
app.config["FREEZER_RELATIVE_URLS"] = True
app.config["FREEZER_STATIC_IGNORE"] = ["*.html"]  # templates live in static/ too for dev
app.config["FREEZER_IGNORE_MIMETYPE_WARNINGS"] = True


# --- Pre-load data ---

def load_songs():
    SOURCE_FILES = {
        "baseline": "baseline.jsonl",
        "prompt_engineered": "prompt_engineered.jsonl",
        "finetuned_original": "finetuned.jsonl",
        "finetuned_gentle": "finetuned_gentle.jsonl",
        "finetuned_wide_attn": "finetuned_wide-attn.jsonl",
        "real": "real.jsonl",
    }
    songs = []
    for source, filename in SOURCE_FILES.items():
        path = OUTPUTS_DIR / filename
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                song = json.loads(line)
                song["source"] = source
                songs.append(song)
    return songs


_SONGS = load_songs()

with open(PAIRS_PATH) as f:
    _ALL_PAIRS = json.load(f)

ACTIVE_APPROACHES = {"finetuned", "prompt_engineered"}
_PAIRS = [
    p for p in _ALL_PAIRS
    if {p["left_approach"], p["right_approach"]} == ACTIVE_APPROACHES
]


# --- Routes (trailing slashes → Frozen-Flask creates dir/index.html) ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/browse/")
def browse():
    return render_template("browse.html")


@app.route("/leaderboard/")
def leaderboard():
    return render_template("leaderboard.html")


# JSON endpoints with .json extension for proper GitHub Pages MIME types
@app.route("/api/songs.json")
def api_songs():
    return jsonify(_SONGS)


@app.route("/api/pairs.json")
def api_pairs():
    blind = []
    for i, p in enumerate(_PAIRS):
        blind.append({
            "index": i,
            "genre": p["genre"],
            "theme": p["theme"],
            "left_lyrics": p["left_lyrics"],
            "right_lyrics": p["right_lyrics"],
        })
    return jsonify(blind)


@app.route("/api/results.json")
def api_results():
    return jsonify({"leaderboard": [], "total_judgments": 0, "blocks": []})


# --- Freeze ---

freezer = Freezer(app)

if __name__ == "__main__":
    freezer.freeze()
    print(f"\nFrozen to {PROJECT_DIR / 'docs'}/")
