"""Crowdsourced Lyrics ELO — Flask app.

Run locally:  python app.py
Production:   gunicorn -b 0.0.0.0:8080 app:app
"""

import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request

app = Flask(__name__, template_folder="static", static_folder="static")

PAIRS_PATH = Path(__file__).resolve().parent / "pairs.json"
DB_PATH = Path(__file__).resolve().parent / "judgments.db"
K = 32
INITIAL_ELO = 1500

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


# --- ELO ---

def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


def update_elo(ra: float, rb: float, result: str) -> tuple[float, float]:
    ea = expected_score(ra, rb)
    eb = 1 - ea
    if result == "A wins":
        sa, sb = 1.0, 0.0
    elif result == "B wins":
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5
    return ra + K * (sa - ea), rb + K * (sb - eb)


def compute_results(db):
    rows = db.execute("SELECT left_approach, right_approach, rating FROM judgments").fetchall()

    elos = defaultdict(lambda: INITIAL_ELO)
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)

    for row in rows:
        a, b, rating = row["left_approach"], row["right_approach"], row["rating"]
        elos[a], elos[b] = update_elo(elos[a], elos[b], rating)
        if rating == "A wins":
            wins[a] += 1
            losses[b] += 1
        elif rating == "B wins":
            wins[b] += 1
            losses[a] += 1
        else:
            ties[a] += 1
            ties[b] += 1

    approaches = sorted(set(elos.keys()), key=lambda x: -elos[x])
    leaderboard = []
    for ap in approaches:
        leaderboard.append({
            "approach": ap,
            "elo": round(elos[ap]),
            "wins": wins[ap],
            "losses": losses[ap],
            "ties": ties[ap],
        })

    return {"leaderboard": leaderboard, "total_judgments": len(rows)}


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html")


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
