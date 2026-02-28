"""Generate random pairwise matchups for evaluation.

Creates outputs/pairs.jsonl with randomized A/B pairings across all approaches.
Each pair shows two songs for the same genre/theme from different approaches.

Run: uv run python make_pairs.py
"""

import json
import random
from itertools import combinations
from pathlib import Path

OUTPUTS_DIR = Path("outputs")


def load_all_songs() -> dict[str, list[dict]]:
    """Load all approach outputs, keyed by approach name."""
    approaches = {}
    for path in OUTPUTS_DIR.glob("*.jsonl"):
        if path.name in ("judgments.jsonl", "pairs.jsonl"):
            continue
        approach = path.stem
        with open(path) as f:
            songs = [json.loads(line) for line in f if line.strip()]
        if songs:
            approaches[approach] = songs
            print(f"  {approach}: {len(songs)} songs")
    return approaches


def make_pairs(approaches: dict[str, list[dict]], seed: int = 42) -> list[dict]:
    """Generate all pairwise matchups, randomize left/right and order."""
    # Index songs by (genre, theme, approach)
    index = {}
    for approach, songs in approaches.items():
        for song in songs:
            key = (song["genre"], song["theme"])
            index[(key, approach)] = song

    pairs = []
    approach_names = sorted(approaches.keys())

    for a, b in combinations(approach_names, 2):
        # Find common genre/theme keys
        keys_a = {(s["genre"], s["theme"]) for s in approaches[a]}
        keys_b = {(s["genre"], s["theme"]) for s in approaches[b]}
        common = sorted(keys_a & keys_b)

        for genre, theme in common:
            song_a = index[((genre, theme), a)]
            song_b = index[((genre, theme), b)]
            pairs.append({
                "genre": genre,
                "theme": theme,
                "left_approach": a,
                "right_approach": b,
                "left_lyrics": song_a["lyrics"],
                "right_lyrics": song_b["lyrics"],
            })

    # Shuffle everything
    rng = random.Random(seed)
    rng.shuffle(pairs)

    # Randomly swap left/right for each pair
    for pair in pairs:
        if rng.random() > 0.5:
            pair["left_approach"], pair["right_approach"] = pair["right_approach"], pair["left_approach"]
            pair["left_lyrics"], pair["right_lyrics"] = pair["right_lyrics"], pair["left_lyrics"]

    return pairs


def main():
    print("Loading songs...")
    approaches = load_all_songs()
    print(f"\n{len(approaches)} approaches loaded")

    pairs = make_pairs(approaches)
    print(f"\n{len(pairs)} pairwise matchups generated")

    # Show breakdown
    from collections import Counter
    matchup_counts = Counter()
    for p in pairs:
        key = tuple(sorted([p["left_approach"], p["right_approach"]]))
        matchup_counts[key] += 1
    for (a, b), count in sorted(matchup_counts.items()):
        print(f"  {a} vs {b}: {count}")

    output_path = OUTPUTS_DIR / "pairs.jsonl"
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
