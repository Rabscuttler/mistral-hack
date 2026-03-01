# Automated Self-Improvement Loop for Song Lyrics Generation

## Context

We need to build a **prompt optimization loop** that demonstrates an AI coding agent (Claude Code) automatically evaluating, analyzing, and improving song lyrics generation using **W&B Weave Evaluations** and **W&B MCP tools**. This is for a hackathon challenge requiring: automated evals, an optimization loop, smart W&B MCP delegation, measurable metric increase, and generated skills/configs as artifacts.

Fine-tuning is too slow for iteration, so the loop optimizes **prompts** — the system prompts, structural instructions, and few-shot examples in `prompts.py` — using Weave Evaluations as the scoring mechanism and Mistral as both the generator and the meta-optimizer.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────┐
  │                  ITERATION N                         │
  │                                                      │
  │  1. Load prompts_v{N}.json                           │
  │  2. Build PromptEngineeredLyricsModel                │
  │  3. Run weave.Evaluation (12 songs × 5 scorers)     │
  │  4. Query results via W&B MCP tools                  │
  │  5. Mistral meta-optimizer rewrites weakest prompts  │
  │  6. Save prompts_v{N+1}.json → loop back to 1       │
  └─────────────────────────────────────────────────────┘

  After all iterations:
  7. Query all evals via MCP, build comparison table
  8. Create W&B Report via create_wandb_report_tool
```

---

## Reference Statistics (10,000 Real Songs vs Generated)

Analyzed 10,000 songs from `data/processed/val.parquet` (204K total) for robust distributions:

**Reference distributions (n=9,991 real songs):**
```
Metric              mean    std     p25     p50     p75
Word count          221     117     140     202     278
Unique word ratio   0.468   0.144   0.364   0.453   0.565
Line count          34      17      22      32      44
Repeat fraction     0.138   0.092   0.067   0.138   0.200
Rhyme rate          0.162   0.161   0.038   0.125   0.241
Contraction rate    0.050   0.041   0.016   0.045   0.075
```

**Per-genre reference (n=3,000 each):**
```
Genre      words  unique_ratio  repeat_frac  rhyme_rate
pop        223    0.458         0.140        0.165
country    232    0.461         0.138        0.163
rock       204    0.494         0.132        0.159
```

**Current generated lyrics vs reference:**
```
                     words  unique_ratio  lines  repeat_frac  rhyme_rate
real (10k avg)       221    0.468         34     0.138        0.162
baseline             394    0.57          53.1   0.14         0.13
prompt_engineered    498    0.56          61.7   0.14         0.13
finetuned            237    0.30          34.5   0.20         0.24
```

**Key gaps:**
- **Too wordy** — baseline/PE are 78-126% longer than real songs (394-498 vs 221 words)
- **Vocabulary too diverse** — unique_ratio 0.56 vs real's 0.47. Real songs repeat words for catchiness
- **Under-rhyming** — baseline/PE 0.13 vs real's 0.16
- **Contractions too rare** — LLMs write formally ("I am" not "I'm")

---

## Implementation Steps

### Step 1: Create `scorers.py` — 5 Custom Weave Scorers

**Design philosophy**: Score by **closeness to real song distributions**, not by maximizing any single metric. A song that's too wordy, too varied, or too rhyme-heavy is penalized just like one that's too short or repetitive.

All extend `weave.Scorer`, implement `score(self, output, ...)` decorated with `@weave.op`.

| Scorer | Type | What it measures | Returns |
|--------|------|-----------------|---------|
| `NaturalnessScorer` | Deterministic | Closeness to real-song distributions: word count, unique word ratio, line count, repetition fraction. Penalizes deviation in either direction. | `naturalness_score` (0-1), individual deltas |
| `RhymeScorer` | Deterministic | End-rhyme density via suffix matching. Target is ~0.16 (real average). Too much or too little rhyming penalized. | `rhyme_score` (0-1), `rhyme_rate` |
| `LLMJudgeScorer` | LLM (Mistral) | Rates emotional_impact, singability, originality, genre_fit, overall (1-10 each) + free-text **critique** identifying specific weaknesses | 5 numeric scores + `critique` string |
| `GenreClassifierScorer` | LLM (Mistral) | Blind genre classification — does the song read as its intended genre? | `genre_match` (bool), `predicted_genre` |
| `AuthenticityScorer` | Deterministic | Checks for LLM-isms: excessive section labels, overly formal language, meta-commentary about songs, missing contractions | `authenticity_score` (0-1) |

**`NaturalnessScorer` detail**: Uses reference distributions from 10K real songs (`data/processed/val.parquet`). Hardcoded reference stats (computed at build time, not at runtime): `word_count: μ=221, σ=117`, `unique_ratio: μ=0.468, σ=0.144`, `line_count: μ=34, σ=17`, `repeat_frac: μ=0.138, σ=0.092`, `contraction_rate: μ=0.050, σ=0.041`. Each metric gets a 0-1 subscore: `exp(-0.5 * ((value - μ) / σ)²)`. Overall naturalness = average of subscores. **Core innovation** — scoring by "does this match real song statistics?" not "is this generically good?"

**`AuthenticityScorer` detail**: Catches common LLM tells — songs that say "This song captures the feeling of...", use overly formal vocabulary, never use contractions ("I am" instead of "I'm"), or have more section headers than actual content lines.

The **LLMJudgeScorer critique text** is the key signal that drives prompt improvement — it identifies *specific* weaknesses for the meta-optimizer to address.

**Key files**: `models.py` (weave.Model interface), weave Scorer base class

### Step 2: Create `eval_loop.py` — Orchestration Script

Core functions:

- **`build_dataset(sample_size=3)`** — Samples 3 themes per genre (12 total) for fast iteration. Uses fixed seed per iteration for reproducibility.
- **`build_model(prompts_config)`** — Constructs `PromptEngineeredLyricsModel` from a prompts dict
- **`run_evaluation(model, iteration, dataset)`** — Runs `weave.Evaluation` with all 5 scorers, named `lyrics-eval-v{iteration}`
- **`analyze_results(results)`** — Extracts per-genre averages, identifies weakest genre and weakest dimension, collects LLM critiques
- **`improve_prompts(analysis, current_config, iteration)`** — Calls Mistral as meta-optimizer (see below)
- **`main_loop(n_iterations=5)`** — Orchestrates everything, saves artifacts after each iteration

**Meta-optimizer prompt** (the self-improvement core): Takes the weakest genre's current system prompt + eval scores + collected critiques → asks Mistral to rewrite the prompt addressing the specific weaknesses. Each iteration targets the single weakest genre to keep changes focused and measurable.

### Step 3: Add Prompt Versioning to `prompts.py`

Add two functions:
- **`get_prompts_config()`** — Returns current prompts as a serializable dict: `{system_prompts, structural_instructions, few_shot_examples}`
- **`load_prompts_config(path)`** — Loads from a JSON artifact file

The existing constants (`GENRE_SYSTEM_PROMPTS`, `STRUCTURAL_INSTRUCTIONS`, `FEW_SHOT_EXAMPLES`) stay unchanged as the v0 baseline.

### Step 4: Create `artifacts/` Directory

Each iteration saves:
- `artifacts/prompts_v{N}.json` — Full prompt config snapshot
- `artifacts/results_v{N}.json` — Evaluation summary (scores per genre per dimension)
- `artifacts/changelog.json` — Append-only log: `{iteration, weakest_genre, weakest_dimension, changes_made, scores_before, scores_after}`

### Step 5: W&B MCP Integration Points

These are used **during** the loop by Claude Code (or called programmatically):

| When | MCP Tool | Purpose |
|------|----------|---------|
| After each eval | `query_weave_traces_tool` | Fetch evaluation results, per-song scores, LLM critiques |
| After each eval | `count_weave_traces_tool` | Track total evals/traces across iterations |
| After all iterations | `query_weave_traces_tool` | Compare metrics across all iterations |
| After all iterations | `create_wandb_report_tool` | Publish a W&B Report with the improvement journey |
| Discovery | `query_wandb_entity_projects` | Verify project exists and get entity name |

### Step 6: Run the Loop (5 iterations)

**Per iteration (~5-8 min):**
1. Generate 12 songs (~2 min)
2. Score with 5 scorers (~3 min, 2 LLM scorers dominate)
3. Analyze + generate improved prompts (~1 min)
4. Save artifacts

**Total: ~30-40 minutes for 5 iterations.**

### Step 7: Create W&B Report

Use `create_wandb_report_tool` to publish a report containing:
- Iteration-over-iteration score table
- Which genres improved most and why
- The prompt evolution (diff of each version)
- Key insights about what prompt changes had the biggest impact

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scorers.py` | **CREATE** | 5 Weave Scorer classes (Naturalness, Rhyme, LLMJudge, GenreClassifier, Authenticity) |
| `eval_loop.py` | **CREATE** | Main loop orchestration |
| `prompts.py` | **MODIFY** | Add `get_prompts_config()` and `load_prompts_config()` |
| `artifacts/` | **CREATE DIR** | Versioned prompt snapshots, results, changelog |

No changes needed to `models.py`, `config.py`, `generate.py` — they already have the right interfaces.

---

## Hackathon Deliverables Checklist

- **Automated evals creation**: 5 custom Weave Scorers, `weave.Evaluation` runs
- **Optimization loop**: 5 iterations of eval → analyze → improve → re-eval
- **Smart delegation**: W&B MCP tools query results, count traces, create reports
- **Measurable metric increase**: Numeric scores tracked per iteration in Weave + artifacts
- **Generated skills/prompts/configs**: `artifacts/prompts_v{0-4}.json`, `scorers.py`, `changelog.json`
- **Creative**: Mistral critiques its own lyrics then rewrites its own instructions
- **End-to-end**: eval → analysis → improvement, fully traced in Weave

---

## Verification

1. Run `python eval_loop.py` — should complete 5 iterations, printing scores each round
2. Check Weave UI: 5 evaluation runs visible under `mistral-hackathon` project
3. Check `artifacts/` — 5 prompt versions, 5 result snapshots, changelog
4. Use W&B MCP `query_weave_traces_tool` to confirm traces are queryable
5. Use W&B MCP `create_wandb_report_tool` to publish the final report
6. Verify overall scores increase from iteration 0 → 4
