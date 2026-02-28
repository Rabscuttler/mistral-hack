"""Prepare the lyrics dataset for fine-tuning.

Extracts the zip, filters by genre/quality, deduplicates, and splits train/val.
"""

import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GENRE_MAP, MAX_LYRICS_LINES, MIN_LYRICS_LINES

DATASET_ZIP = Path(os.path.expanduser("~/Downloads/genius-dataset.zip"))
EXTRACT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def extract_zip():
    """Extract the dataset zip if not already done."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXTRACT_DIR / "song_lyrics.csv"
    if csv_path.exists():
        print(f"Already extracted: {csv_path}")
        return csv_path

    print(f"Extracting {DATASET_ZIP}...")
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")
    return csv_path


def explore_schema(csv_path: Path):
    """Print dataset schema and basic stats."""
    df_sample = pd.read_csv(csv_path, nrows=5)
    print("Columns:", list(df_sample.columns))
    print("Sample:\n", df_sample.head())
    print("\nDtypes:\n", df_sample.dtypes)
    return list(df_sample.columns)


def filter_and_clean(csv_path: Path, chunk_size: int = 100_000) -> pd.DataFrame:
    """Read CSV in chunks, filter to target genres, English, and quality thresholds."""
    target_genres = set(GENRE_MAP.values())
    chunks = []

    print("Reading and filtering dataset in chunks...")
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        # Filter to target genres (case-insensitive match)
        if "tag" in chunk.columns:
            genre_col = "tag"
        elif "genre" in chunk.columns:
            genre_col = "genre"
        else:
            print(f"Available columns: {list(chunk.columns)}")
            raise ValueError("Cannot find genre/tag column")

        chunk = chunk[chunk[genre_col].isin(target_genres)]

        # Filter to English if language column exists
        if "language" in chunk.columns:
            chunk = chunk[chunk["language"].str.lower() == "en"]

        # Drop rows with missing lyrics
        lyrics_col = "lyrics" if "lyrics" in chunk.columns else None
        if lyrics_col is None:
            for col in chunk.columns:
                if "lyric" in col.lower():
                    lyrics_col = col
                    break
        if lyrics_col is None:
            raise ValueError(f"Cannot find lyrics column. Columns: {list(chunk.columns)}")

        chunk = chunk.dropna(subset=[lyrics_col])

        # Quality filter: line count
        line_counts = chunk[lyrics_col].str.count("\n") + 1
        chunk = chunk[
            (line_counts >= MIN_LYRICS_LINES) & (line_counts <= MAX_LYRICS_LINES)
        ]

        chunks.append(chunk)
        print(f"  Chunk {i}: {len(chunk)} rows after filtering")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total after filtering: {len(df)} rows")

    # Deduplicate by lyrics content
    lyrics_col_name = lyrics_col
    before = len(df)
    df = df.drop_duplicates(subset=[lyrics_col_name])
    print(f"After dedup: {len(df)} rows (removed {before - len(df)} duplicates)")

    return df


def split_and_save(df: pd.DataFrame):
    """90/10 train/val split and save to parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 90/10 split
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    train_path = OUTPUT_DIR / "train.parquet"
    val_path = OUTPUT_DIR / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Val:   {len(val_df)} rows -> {val_path}")

    # Print genre distribution
    genre_col = "tag" if "tag" in df.columns else "genre"
    print(f"\nGenre distribution (train):\n{train_df[genre_col].value_counts()}")

    return train_path, val_path


if __name__ == "__main__":
    csv_path = extract_zip()
    explore_schema(csv_path)
    df = filter_and_clean(csv_path)
    split_and_save(df)
