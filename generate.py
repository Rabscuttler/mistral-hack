"""Generate song lyrics using different approaches, all traced in Weave."""

import json
import os
from pathlib import Path

import weave
from dotenv import load_dotenv

from config import BASELINE_MODEL, GENRES, THEMES, WEAVE_PROJECT
from models import MistralLyricsModel, PromptEngineeredLyricsModel
from prompts import FEW_SHOT_EXAMPLES, GENRE_SYSTEM_PROMPTS, STRUCTURAL_INSTRUCTIONS

load_dotenv()

OUTPUTS_DIR = Path("outputs")


def generate_all(model: weave.Model, approach: str) -> list[dict]:
    """Generate lyrics for all genre/theme combos using the given model."""
    results = []
    for genre in GENRES:
        for theme in THEMES[genre]:
            print(f"[{approach}] Generating {genre}: {theme}...")
            result = model.predict(genre=genre, theme=theme)
            result["approach"] = approach
            results.append(result)
    return results


def save_results(results: list[dict], approach: str) -> Path:
    """Save generation results to a JSONL file."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / f"{approach}.jsonl"
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {path}")
    return path


def run_baseline():
    """Step 1: Generate lyrics with minimal prompting."""
    model = MistralLyricsModel(model_name=BASELINE_MODEL)
    results = generate_all(model, "baseline")
    save_results(results, "baseline")
    return results


def run_prompt_engineered():
    """Step 3: Generate lyrics with enhanced prompting."""
    model = PromptEngineeredLyricsModel(
        model_name=BASELINE_MODEL,
        system_prompts=GENRE_SYSTEM_PROMPTS,
        structural_instructions=STRUCTURAL_INSTRUCTIONS,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )
    results = generate_all(model, "prompt_engineered")
    save_results(results, "prompt_engineered")
    return results


def run_finetuned(model_name: str):
    """Step 2c: Generate lyrics with fine-tuned model."""
    model = MistralLyricsModel(model_name=model_name)
    results = generate_all(model, "finetuned")
    save_results(results, "finetuned")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate song lyrics")
    parser.add_argument(
        "approach",
        choices=["baseline", "prompt_engineered", "finetuned", "all"],
        help="Which generation approach to run",
    )
    parser.add_argument(
        "--finetuned-model",
        default=None,
        help="Model name/path for fine-tuned generation",
    )
    args = parser.parse_args()

    weave.init(WEAVE_PROJECT)

    if args.approach == "baseline":
        run_baseline()
    elif args.approach == "prompt_engineered":
        run_prompt_engineered()
    elif args.approach == "finetuned":
        if not args.finetuned_model:
            parser.error("--finetuned-model required for finetuned approach")
        run_finetuned(args.finetuned_model)
    elif args.approach == "all":
        run_baseline()
        run_prompt_engineered()
        if args.finetuned_model:
            run_finetuned(args.finetuned_model)
        else:
            print("Skipping finetuned (no --finetuned-model provided)")
