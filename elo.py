"""ELO rating system for song lyrics evaluation.

Reads pairwise judgments from outputs/judgments.jsonl and computes
ELO ratings per approach (and optionally per approach+genre).

Run: uv run python elo.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path("outputs/judgments.jsonl")
K = 32  # Standard ELO K-factor
INITIAL_ELO = 1500


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


def update_elo(ra: float, rb: float, result: str) -> tuple[float, float]:
    """Update ELO ratings given a result ('A wins', 'B wins', 'Tie').

    Returns (new_ra, new_rb).
    """
    ea = expected_score(ra, rb)
    eb = 1 - ea

    if result == "A wins":
        sa, sb = 1.0, 0.0
    elif result == "B wins":
        sa, sb = 0.0, 1.0
    else:  # Tie
        sa, sb = 0.5, 0.5

    new_ra = ra + K * (sa - ea)
    new_rb = rb + K * (sb - eb)
    return new_ra, new_rb


def load_judgments() -> list[dict]:
    if not RESULTS_FILE.exists():
        print(f"No judgments file at {RESULTS_FILE}")
        return []
    with open(RESULTS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_elo(judgments: list[dict]) -> dict[str, float]:
    """Compute overall ELO per approach."""
    elos = defaultdict(lambda: INITIAL_ELO)

    for j in judgments:
        a = j["left_approach"]
        b = j["right_approach"]
        # Support both nested and flat rating formats
        rating = j.get("rating") or j.get("ratings", {}).get("overall", "Tie")
        elos[a], elos[b] = update_elo(elos[a], elos[b], rating)

    return dict(elos)


def compute_elo_by_genre(judgments: list[dict]) -> dict[str, dict[str, float]]:
    """Compute ELO per approach per genre."""
    elos = defaultdict(lambda: defaultdict(lambda: INITIAL_ELO))

    for j in judgments:
        genre = j["genre"]
        a = j["left_approach"]
        b = j["right_approach"]
        rating = j.get("rating") or j.get("ratings", {}).get("overall", "Tie")
        elos[genre][a], elos[genre][b] = update_elo(elos[genre][a], elos[genre][b], rating)

    return {g: dict(e) for g, e in elos.items()}


def main():
    judgments = load_judgments()
    if not judgments:
        return

    print(f"Loaded {len(judgments)} judgments\n")

    # Overall ELO
    elos = compute_elo(judgments)
    print("=== Overall ELO ===")
    for approach, elo in sorted(elos.items(), key=lambda x: -x[1]):
        print(f"  {approach:25s} {elo:.0f}")

    # Win/loss/tie counts
    print("\n=== Record ===")
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    for j in judgments:
        a = j["left_approach"]
        b = j["right_approach"]
        rating = j.get("rating") or j.get("ratings", {}).get("overall", "Tie")
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
    for app in all_approaches:
        w, l, t = wins[app], losses[app], ties[app]
        total = w + l + t
        print(f"  {app:25s} {w}W {l}L {t}T ({total} games)")

    # Per-genre ELO
    by_genre = compute_elo_by_genre(judgments)
    if by_genre:
        print("\n=== ELO by Genre ===")
        for genre in sorted(by_genre):
            print(f"\n  {genre}:")
            for approach, elo in sorted(by_genre[genre].items(), key=lambda x: -x[1]):
                print(f"    {approach:23s} {elo:.0f}")


if __name__ == "__main__":
    main()
