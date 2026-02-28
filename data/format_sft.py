"""Format the processed dataset as conversational JSONL and upload to HuggingFace Hub."""

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_DATASET_REPO

PROCESSED_DIR = Path("data/processed")
SFT_DIR = Path("data/sft")


def format_conversation(row: dict) -> dict:
    """Convert a row to conversational format for SFT."""
    genre_col = "tag" if "tag" in row else "genre"
    genre = row[genre_col]
    title = row.get("title", "Untitled")
    lyrics_col = "lyrics" if "lyrics" in row else None
    if lyrics_col is None:
        for key in row:
            if "lyric" in key.lower():
                lyrics_col = key
                break
    lyrics = row[lyrics_col]

    return {
        "messages": json.dumps([
            {
                "role": "user",
                "content": f"Write a {genre.lower()} song titled '{title}'",
            },
            {"role": "assistant", "content": lyrics},
        ])
    }


def format_split(parquet_path: Path) -> list[dict]:
    """Format a parquet split into conversational JSONL."""
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        records.append(format_conversation(row.to_dict()))
    return records


def save_jsonl(records: list[dict], output_path: Path):
    """Save records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records to {output_path}")


def upload_to_hub(repo_id: str, max_train: int = 50_000, max_val: int = 5_000):
    """Upload a subsampled SFT dataset to HuggingFace Hub.

    Full dataset is 1.8M+ rows which is too large for LoRA SFT.
    Subsample to max_train/max_val for tractable training.
    """
    import random

    train_path = SFT_DIR / "train.jsonl"
    val_path = SFT_DIR / "val.jsonl"

    def load_jsonl_sampled(path, max_rows):
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        if len(records) > max_rows:
            random.seed(42)
            records = random.sample(records, max_rows)
        print(f"  Loaded {len(records)} records from {path}")
        return Dataset.from_list(records)

    print(f"Loading and subsampling (train={max_train}, val={max_val})...")
    ds = DatasetDict({
        "train": load_jsonl_sampled(train_path, max_train),
        "validation": load_jsonl_sampled(val_path, max_val),
    })

    print(f"Uploading to {repo_id}...")
    ds.push_to_hub(repo_id, private=False)
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--repo-id", default=HF_DATASET_REPO, help="HF dataset repo ID")
    args = parser.parse_args()

    print("Formatting train split...")
    train_records = format_split(PROCESSED_DIR / "train.parquet")
    save_jsonl(train_records, SFT_DIR / "train.jsonl")

    print("Formatting val split...")
    val_records = format_split(PROCESSED_DIR / "val.parquet")
    save_jsonl(val_records, SFT_DIR / "val.jsonl")

    if args.upload:
        upload_to_hub(args.repo_id)
