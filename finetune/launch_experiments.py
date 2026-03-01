"""Launch two experimental fine-tuning jobs on HuggingFace Jobs."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import run_uv_job

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_DATASET_REPO, HF_USERNAME

load_dotenv()

SFT_SCRIPT = Path(__file__).parent / "sft_script.py"

EXPERIMENTS = {
    "gentle": {
        "description": "Run A: Gentle — same LoRA shape, 10x lower LR, more warmup/dropout",
        "hub_model_id": f"{HF_USERNAME}/mistral-7b-lyrics-lora-gentle",
        "env": {
            "DATASET_REPO": HF_DATASET_REPO,
            "WANDB_PROJECT": "mistral-hackathon",
            "WANDB_RUN_NAME": "lyrics-sft-gentle",
            "NUM_EPOCHS": "3",
            "LEARNING_RATE": "2e-5",
            "LORA_R": "16",
            "LORA_ALPHA": "32",
            "LORA_DROPOUT": "0.1",
            "WARMUP_RATIO": "0.1",
            "MAX_TRAIN_SAMPLES": "250000",
            "BATCH_SIZE": "4",
            "GRADIENT_ACCUMULATION": "8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # TARGET_MODULES omitted = default all modules
        },
    },
    "wide-attn": {
        "description": "Run B: Wide Attention — r=64 on attention layers only",
        "hub_model_id": f"{HF_USERNAME}/mistral-7b-lyrics-lora-wide-attn",
        "env": {
            "DATASET_REPO": HF_DATASET_REPO,
            "WANDB_PROJECT": "mistral-hackathon",
            "WANDB_RUN_NAME": "lyrics-sft-wide-attn",
            "NUM_EPOCHS": "3",
            "LEARNING_RATE": "5e-5",
            "LORA_R": "64",
            "LORA_ALPHA": "128",
            "LORA_DROPOUT": "0.1",
            "WARMUP_RATIO": "0.03",
            "MAX_TRAIN_SAMPLES": "250000",
            "BATCH_SIZE": "2",
            "GRADIENT_ACCUMULATION": "16",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TARGET_MODULES": "q_proj,k_proj,v_proj,o_proj",
        },
    },
}


def launch(name: str, flavor: str = "a10g-large", timeout: str = "4h"):
    exp = EXPERIMENTS[name]
    env = dict(exp["env"])
    env["HUB_MODEL_ID"] = exp["hub_model_id"]

    print(f"\n{'='*60}")
    print(f"Launching: {exp['description']}")
    print(f"  Model repo: {exp['hub_model_id']}")
    print(f"  Flavor: {flavor}")
    print(f"  Timeout: {timeout}")
    for k, v in sorted(env.items()):
        if "KEY" not in k and "TOKEN" not in k:
            print(f"  {k}={v}")
    print(f"{'='*60}")

    job = run_uv_job(
        script=str(SFT_SCRIPT),
        flavor=flavor,
        env=env,
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

    parser = argparse.ArgumentParser(description="Launch experimental SFT jobs")
    parser.add_argument(
        "experiments",
        nargs="*",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Which experiments to launch (default: all)",
    )
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--timeout", default="4h")
    args = parser.parse_args()

    jobs = {}
    for name in args.experiments:
        jobs[name] = launch(name, flavor=args.flavor, timeout=args.timeout)

    print(f"\n{'='*60}")
    print(f"All {len(jobs)} jobs launched!")
    for name, job in jobs.items():
        print(f"  {name}: {job}")
    print(f"{'='*60}")
