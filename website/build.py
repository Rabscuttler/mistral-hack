"""Convert outputs/pairs.jsonl to website/pairs.json.

Run: python website/build.py
"""

import json
from pathlib import Path

pairs_jsonl = Path(__file__).resolve().parent.parent / "outputs" / "pairs.jsonl"
pairs_json = Path(__file__).resolve().parent / "pairs.json"

with open(pairs_jsonl) as f:
    pairs = [json.loads(line) for line in f if line.strip()]

with open(pairs_json, "w") as f:
    json.dump(pairs, f)

print(f"Wrote {len(pairs)} pairs to {pairs_json}")
