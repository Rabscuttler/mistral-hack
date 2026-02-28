"""Sample real songs from the dataset as a control set for evaluation.

Picks 10 songs per genre from the validation set, preferring popular songs.
Outputs in the same format as generate.py for use in judging.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import GENRES, GENRE_MAP, THEMES

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")


def sample_real_songs():
    df = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    # Filter to reasonable quality: has verse/chorus structure, decent length
    df = df[df["lyrics"].str.contains(r"\[Verse", case=False, na=False)]
    df["line_count"] = df["lyrics"].str.count("\n") + 1
    df = df[(df["line_count"] >= 10) & (df["line_count"] <= 100)]

    results = []
    for genre in GENRES:
        tag = GENRE_MAP[genre]
        genre_df = df[df["tag"] == tag].copy()

        # Sort by views (popularity) and take top 500, then sample 10
        genre_df = genre_df.sort_values("views", ascending=False).head(500)

        # Sample 10 songs, pair each with a theme from our config
        themes = THEMES[genre]
        sampled = genre_df.sample(n=min(10, len(genre_df)), random_state=42)

        for i, (_, row) in enumerate(sampled.iterrows()):
            theme = themes[i] if i < len(themes) else themes[0]
            results.append({
                "lyrics": row["lyrics"],
                "genre": genre,
                "theme": theme,
                "model": "real",
                "approach": "real",
                "title": row["title"],
                "artist": row["artist"],
            })

    # Save
    OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path = OUTPUTS_DIR / "real.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} real songs to {output_path}")
    for genre in GENRES:
        count = sum(1 for r in results if r["genre"] == genre)
        print(f"  {genre}: {count}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sample_real_songs()
