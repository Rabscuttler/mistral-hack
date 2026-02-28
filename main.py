"""Song Lyrics Pipeline - Entry point.

Usage:
    # Step 1: Generate baseline lyrics
    uv run generate.py baseline

    # Step 2a: Prepare dataset
    uv run data/prepare.py

    # Step 2a: Format for SFT and optionally upload
    uv run data/format_sft.py --upload

    # Step 2b: Launch fine-tuning on HF Jobs
    uv run finetune/train_hf_jobs.py

    # Step 2c: Generate with fine-tuned model
    uv run generate.py finetuned --finetuned-model your-hf-username/mistral-7b-lyrics-lora

    # Step 3: Generate with prompt engineering
    uv run generate.py prompt_engineered

    # Step 4: Judge lyrics
    streamlit run judge.py
"""

import weave
from dotenv import load_dotenv

from config import WEAVE_PROJECT

load_dotenv()


def main():
    weave.init(WEAVE_PROJECT)
    print("Song Lyrics Pipeline initialized.")
    print("Run individual steps with the commands listed in this file's docstring.")
    print(f"Weave project: {WEAVE_PROJECT}")


if __name__ == "__main__":
    main()
