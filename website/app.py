"""Crowdsourced Lyrics ELO — Flask app.

Run locally:  python app.py
Production:   gunicorn -b 0.0.0.0:8080 app:app
"""

import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

PAIRS_PATH = Path(__file__).resolve().parent / "pairs.json"
DB_PATH = Path(__file__).resolve().parent / "judgments.db"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Load pairs into memory at startup
with open(PAIRS_PATH) as f:
    _ALL_PAIRS = json.load(f)

# Filter to only finetuned vs prompt_engineered for now
ACTIVE_APPROACHES = {"finetuned", "prompt_engineered"}
PAIRS = [p for p in _ALL_PAIRS if {p["left_approach"], p["right_approach"]} == ACTIVE_APPROACHES]


# --- Database ---

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair_index INTEGER NOT NULL,
            genre TEXT NOT NULL,
            theme TEXT NOT NULL,
            left_approach TEXT NOT NULL,
            right_approach TEXT NOT NULL,
            rating TEXT NOT NULL,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_session ON judgments(session_id)")
    db.commit()

    # Import existing judgments.jsonl on first run
    count = db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
    if count == 0:
        legacy = Path(__file__).resolve().parent.parent / "outputs" / "judgments.jsonl"
        if legacy.exists():
            imported = 0
            with open(legacy) as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    # Find matching pair index
                    for i, p in enumerate(PAIRS):
                        if (p["genre"] == j["genre"]
                                and p["left_approach"] == j["left_approach"]
                                and p["right_approach"] == j["right_approach"]):
                            db.execute(
                                "INSERT INTO judgments (pair_index, genre, theme, left_approach, right_approach, rating, session_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (i, j["genre"], j.get("theme", ""), j["left_approach"], j["right_approach"], j["rating"], "legacy", j.get("timestamp", time.time())),
                            )
                            imported += 1
                            break
            db.commit()
            print(f"Imported {imported} legacy judgments")

    db.close()


# --- Scoring ---

def _tally(rows):
    """Compute win rate and W/L/T from a list of judgment rows."""
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)

    for row in rows:
        a, b, rating = row["left_approach"], row["right_approach"], row["rating"]
        if rating == "A wins":
            wins[a] += 1
            losses[b] += 1
        elif rating == "B wins":
            wins[b] += 1
            losses[a] += 1
        else:
            ties[a] += 1
            ties[b] += 1

    all_approaches = sorted(set(wins) | set(losses) | set(ties))

    def win_rate(ap):
        total = wins[ap] + losses[ap] + ties[ap]
        if total == 0:
            return 0
        return (wins[ap] + 0.5 * ties[ap]) / total

    all_approaches.sort(key=win_rate, reverse=True)
    return [{
        "approach": ap,
        "win_rate": round(win_rate(ap) * 100),
        "wins": wins[ap],
        "losses": losses[ap],
        "ties": ties[ap],
    } for ap in all_approaches]


def compute_results(db):
    rows = db.execute("SELECT left_approach, right_approach, rating, session_id, timestamp FROM judgments ORDER BY timestamp").fetchall()

    # All-time leaderboard
    leaderboard = _tally(rows)

    # Blocks of 10, most recent first
    reversed_rows = list(reversed(rows))
    blocks = []
    for i in range(0, len(reversed_rows), 10):
        chunk = reversed_rows[i:i + 10]
        blocks.append({
            "count": len(chunk),
            "results": _tally(chunk),
        })

    return {"leaderboard": leaderboard, "total_judgments": len(rows), "blocks": blocks}


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html")


@app.route("/browse")
def browse():
    return render_template("browse.html")


@app.route("/api/pairs")
def api_pairs():
    # Strip approach names for blind evaluation
    blind = []
    for i, p in enumerate(PAIRS):
        blind.append({
            "index": i,
            "genre": p["genre"],
            "theme": p["theme"],
            "left_lyrics": p["left_lyrics"],
            "right_lyrics": p["right_lyrics"],
        })
    return jsonify(blind)


@app.route("/api/judge", methods=["POST"])
def api_judge():
    data = request.get_json()
    pair_index = data.get("pair_index")
    rating = data.get("rating")
    session_id = data.get("session_id")

    if pair_index is None or rating not in ("A wins", "B wins", "Tie") or not session_id:
        return jsonify({"error": "Invalid request"}), 400

    if pair_index < 0 or pair_index >= len(PAIRS):
        return jsonify({"error": "Invalid pair index"}), 400

    pair = PAIRS[pair_index]
    db = get_db()
    db.execute(
        "INSERT INTO judgments (pair_index, genre, theme, left_approach, right_approach, rating, session_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pair_index, pair["genre"], pair["theme"], pair["left_approach"], pair["right_approach"], rating, session_id, time.time()),
    )
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/results")
def api_results():
    db = get_db()
    return jsonify(compute_results(db))


@app.route("/api/songs")
def api_songs():
    """Load all generated songs from outputs/ directory."""
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
    return jsonify(songs)


@app.route("/api/progress/<session_id>")
def api_progress(session_id):
    db = get_db()
    rows = db.execute("SELECT DISTINCT pair_index FROM judgments WHERE session_id = ?", (session_id,)).fetchall()
    return jsonify({"judged": [row["pair_index"] for row in rows]})


# --- Main ---

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
else:
    init_db()
