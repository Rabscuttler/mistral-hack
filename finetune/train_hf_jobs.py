"""Launch fine-tuning on HuggingFace Jobs via run_uv_job()."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import run_uv_job

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_DATASET_REPO, HF_MODEL_REPO

load_dotenv()

SFT_SCRIPT = Path(__file__).parent / "sft_script.py"


def launch_training(
    dataset_repo: str = HF_DATASET_REPO,
    model_repo: str = HF_MODEL_REPO,
    flavor: str = "a100-large",
    timeout: str = "6h",
):
    """Launch SFT fine-tuning on HuggingFace Jobs."""
    print(f"Launching HF Job...")
    print(f"  Script: {SFT_SCRIPT}")
    print(f"  Dataset: {dataset_repo}")
    print(f"  Output model: {model_repo}")
    print(f"  Flavor: {flavor}")

    job = run_uv_job(
        script=str(SFT_SCRIPT),
        flavor=flavor,
        env={
            "DATASET_REPO": dataset_repo,
            "HUB_MODEL_ID": model_repo,
            "WANDB_PROJECT": "mistral-hackathon",
        },
        secrets={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
        timeout=timeout,
    )

    print(f"Job launched: {job}")
    return job


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch SFT fine-tuning on HF Jobs")
    parser.add_argument("--dataset-repo", default=HF_DATASET_REPO)
    parser.add_argument("--model-repo", default=HF_MODEL_REPO)
    parser.add_argument("--flavor", default="a100-large")
    args = parser.parse_args()

    job = launch_training(
        dataset_repo=args.dataset_repo,
        model_repo=args.model_repo,
        flavor=args.flavor,
    )
