"""Self-improvement loop: evaluate → analyze → improve prompts → repeat.

Uses Weave Evaluations with 5 custom scorers. Mistral acts as both the
lyrics generator and the meta-optimizer that rewrites its own prompts.

Usage:
    source .venv/bin/activate
    python eval_loop.py                # 5 iterations
    python eval_loop.py --iterations 3 # custom count
    python eval_loop.py --sample-size 5 # more songs per genre
"""

import asyncio
import copy
import json
import os
import random
import re
from pathlib import Path

import weave
from dotenv import load_dotenv
from mistralai import Mistral

from config import BASELINE_MODEL, GENRES, THEMES, WEAVE_PROJECT
from models import PromptEngineeredLyricsModel
from prompts import get_prompts_config, save_prompts_config
from scorers import (
    AuthenticityScorer,
    GenreClassifierScorer,
    LLMJudgeScorer,
    NaturalnessScorer,
    RhymeScorer,
)

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")


def build_dataset(sample_size: int = 3, seed: int = 42) -> list[dict]:
    """Build eval dataset — sample_size themes per genre."""
    rng = random.Random(seed)
    dataset = []
    for genre in GENRES:
        sampled = rng.sample(THEMES[genre], min(sample_size, len(THEMES[genre])))
        for theme in sampled:
            dataset.append({"genre": genre, "theme": theme})
    return dataset


def build_model(prompts_config: dict) -> PromptEngineeredLyricsModel:
    """Build a PromptEngineeredLyricsModel from a prompts config dict."""
    return PromptEngineeredLyricsModel(
        model_name=BASELINE_MODEL,
        system_prompts=prompts_config["system_prompts"],
        structural_instructions=prompts_config["structural_instructions"],
        few_shot_examples=prompts_config["few_shot_examples"],
    )


async def run_evaluation(
    model: PromptEngineeredLyricsModel,
    iteration: int,
    dataset: list[dict],
) -> dict:
    """Run a Weave evaluation with all 5 scorers."""
    evaluation = weave.Evaluation(
        name=f"lyrics-eval-v{iteration}",
        dataset=dataset,
        scorers=[
            NaturalnessScorer(),
            RhymeScorer(),
            AuthenticityScorer(),
            LLMJudgeScorer(),
            GenreClassifierScorer(),
        ],
    )
    results = await evaluation.evaluate(model)
    return results


def analyze_results(results: dict, dataset: list[dict]) -> dict:
    """Analyze evaluation results to identify weaknesses.

    Returns a dict with:
      - per_genre: {genre: {metric: avg_score}}
      - weakest_genre: str
      - weakest_dimension: str
      - critiques: list of critique strings
      - summary: str
    """
    # The results dict from weave.Evaluation has scorer summaries
    # We need to extract the per-row details from the results
    analysis = {
        "overall_scores": {},
        "critiques": [],
        "weakest_genre": None,
        "weakest_dimension": None,
        "summary": "",
    }

    # Extract top-level averages from results
    for key, value in results.items():
        if isinstance(value, dict) and "mean" in value:
            analysis["overall_scores"][key] = value["mean"]
        elif isinstance(value, (int, float)):
            analysis["overall_scores"][key] = value

    return analysis


def extract_critiques_and_genre_scores(results: dict) -> tuple[list[str], dict[str, float]]:
    """Extract critiques and per-genre overall scores from eval results."""
    critiques = []
    genre_scores = {}

    # Results from weave.Evaluation are summarized — we'll work with what we get
    # The detailed per-row data is in Weave traces, queryable via MCP
    return critiques, genre_scores


def improve_prompts(
    results: dict,
    current_config: dict,
    iteration: int,
) -> dict:
    """Use Mistral as a meta-optimizer to improve prompts based on eval results.

    Targets the weakest-performing area for improvement.
    """
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Build a summary of current scores for the meta-optimizer
    flat = flatten_results(results)
    score_summary = []
    for key, value in sorted(flat.items()):
        if "latency" not in key and "tokens" not in key:
            score_summary.append(f"  {key}: {value:.3f}")

    scores_text = "\n".join(score_summary) if score_summary else "  (no scores available)"

    # Rotate which genre we focus on improving each iteration
    target_genre = GENRES[iteration % len(GENRES)]
    current_system_prompt = current_config["system_prompts"].get(target_genre, "")
    current_structural = current_config["structural_instructions"]

    meta_prompt = f"""You are a prompt engineering expert optimizing song lyrics generation.

CURRENT EVALUATION SCORES (higher is better, scale varies by metric):
{scores_text}

KEY REFERENCE STATS FOR REAL SONGS:
- Real songs average ~221 words (current LLM output is ~400-500, way too long)
- Real songs have unique word ratio ~0.47 (LLM is ~0.56, too diverse — real songs repeat words for catchiness)
- Real songs have repeat fraction ~0.14 (choruses should repeat!)
- Real songs use contractions naturally ("I'm", "don't", "can't")
- Real songs have rhyme rate ~0.16

CURRENT SYSTEM PROMPT FOR {target_genre.upper()}:
{current_system_prompt}

CURRENT STRUCTURAL INSTRUCTIONS:
{current_structural}

Your task: Rewrite BOTH the system prompt for {target_genre} AND the structural instructions to produce more authentic, natural-sounding lyrics. Key improvements needed:
1. Songs should be SHORTER (around 200-250 words, not 400+)
2. Use contractions naturally (I'm, don't, won't, can't, it's)
3. Repeat key phrases in chorus for catchiness — don't try to make every line unique
4. Keep vocabulary simple and conversational, not literary
5. Maintain genre authenticity for {target_genre}

Respond in this exact JSON format:
{{
    "system_prompt": "<the improved system prompt for {target_genre}>",
    "structural_instructions": "<improved structural instructions>",
    "reasoning": "<1-2 sentences explaining what you changed and why>"
}}"""

    import time as _time
    for _attempt in range(5):
        try:
            response = client.chat.complete(
                model="mistral-medium-latest",
                messages=[{"role": "user", "content": meta_prompt}],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=1500,
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** _attempt + 1
                print(f"  [Rate limited] Retrying in {wait}s (attempt {_attempt + 1}/5)")
                _time.sleep(wait)
            else:
                raise
    else:
        response = client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": meta_prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=1500,
        )

    text = response.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        improvements = json.loads(text)
    except json.JSONDecodeError:
        # Mistral sometimes puts literal newlines inside JSON string values
        sanitized = re.sub(r'(?<=": ")(.*?)(?="[,}])', lambda m: m.group(0).replace('\n', '\\n'), text, flags=re.DOTALL)
        try:
            improvements = json.loads(sanitized)
        except json.JSONDecodeError:
            print(f"  [WARN] Could not parse meta-optimizer response, keeping current prompts")
            print(f"  Response: {text[:200]}")
            return current_config

    new_config = copy.deepcopy(current_config)
    new_config["system_prompts"][target_genre] = improvements["system_prompt"]
    new_config["structural_instructions"] = improvements["structural_instructions"]

    reasoning = improvements.get("reasoning", "")
    print(f"  Improved {target_genre} prompt: {reasoning}")

    return new_config


def flatten_results(results: dict, prefix: str = "") -> dict[str, float]:
    """Flatten nested results into a flat dict of metric_name -> value."""
    flat = {}
    for key, value in results.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if "mean" in value:
                flat[full_key] = value["mean"]
            elif "true_fraction" in value:
                flat[full_key] = value["true_fraction"]
            else:
                flat.update(flatten_results(value, full_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[full_key] = value
    return flat


def print_results_summary(results: dict, iteration: int) -> None:
    """Pretty-print evaluation results."""
    flat = flatten_results(results)
    print(f"\n{'='*60}")
    print(f"  ITERATION {iteration} RESULTS")
    print(f"{'='*60}")
    for key, value in sorted(flat.items()):
        if "latency" not in key and "tokens" not in key:
            print(f"  {key:45s} = {value:.3f}")
    print(f"{'='*60}\n")


def main_loop(n_iterations: int = 5, sample_size: int = 3) -> None:
    """Run the self-improvement loop."""
    weave.init(WEAVE_PROJECT)

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Build evaluation dataset (same across iterations for comparability)
    dataset = build_dataset(sample_size=sample_size)
    print(f"Eval dataset: {len(dataset)} songs ({sample_size} per genre)")

    # Load initial prompts (v0)
    prompts_config = get_prompts_config()
    all_results = []
    changelog = []

    for i in range(n_iterations):
        print(f"\n{'#'*60}")
        print(f"  ITERATION {i}/{n_iterations - 1}")
        print(f"  Target genre for improvement: {GENRES[i % len(GENRES)]}")
        print(f"{'#'*60}")

        # Save current prompts
        save_prompts_config(prompts_config, ARTIFACTS_DIR / f"prompts_v{i}.json")

        # Build model and run evaluation
        model = build_model(prompts_config)
        print(f"  Running evaluation...")
        results = asyncio.run(run_evaluation(model, i, dataset))

        # Print and save results
        print_results_summary(results, i)

        # Save results (convert non-serializable types)
        results_serializable = {}
        for k, v in results.items():
            if isinstance(v, (dict, list, str, int, float, bool)):
                results_serializable[k] = v

        with open(ARTIFACTS_DIR / f"results_v{i}.json", "w") as f:
            json.dump(results_serializable, f, indent=2, default=str)

        all_results.append({"iteration": i, "results": results_serializable})

        # Improve prompts for next iteration (skip on last)
        if i < n_iterations - 1:
            print(f"  Improving prompts...")
            old_config = copy.deepcopy(prompts_config)
            prompts_config = improve_prompts(results, prompts_config, i)

            target_genre = GENRES[i % len(GENRES)]
            changelog.append({
                "iteration": i,
                "target_genre": target_genre,
                "old_system_prompt": old_config["system_prompts"][target_genre],
                "new_system_prompt": prompts_config["system_prompts"][target_genre],
                "old_structural": old_config["structural_instructions"],
                "new_structural": prompts_config["structural_instructions"],
            })

    # Save final prompts
    save_prompts_config(prompts_config, ARTIFACTS_DIR / f"prompts_v{n_iterations - 1}_final.json")

    # Save changelog
    with open(ARTIFACTS_DIR / "changelog.json", "w") as f:
        json.dump(changelog, f, indent=2)

    # Save all results for comparison
    with open(ARTIFACTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final comparison
    # Key metrics to track across iterations
    KEY_METRICS = [
        "NaturalnessScorer.naturalness_score",
        "RhymeScorer.rhyme_score",
        "AuthenticityScorer.authenticity_score",
        "LLMJudgeScorer.overall",
        "GenreClassifierScorer.genre_match",
    ]

    print(f"\n{'='*60}")
    print("  IMPROVEMENT SUMMARY")
    print(f"{'='*60}")
    for entry in all_results:
        i = entry["iteration"]
        flat = flatten_results(entry["results"])
        scores = []
        for metric in KEY_METRICS:
            if metric in flat:
                short_name = metric.split(".")[-1]
                scores.append(f"{short_name}={flat[metric]:.3f}")
        print(f"  v{i}: {', '.join(scores)}")
    print(f"{'='*60}")
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run lyrics self-improvement loop")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()

    main_loop(n_iterations=args.iterations, sample_size=args.sample_size)
