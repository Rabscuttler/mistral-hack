# Song Lyrics Pipeline - Plan

## Context

Generate song lyrics three ways (baseline, fine-tuned, prompt-engineered), then compare them with pairwise human evaluation (x > y). Genres: pop, country, rock, indie. Track everything with W&B Weave. Fine-tune via HuggingFace Jobs.

**Dataset**: `~/Downloads/genius-dataset.zip` (3.3GB zip, 9GB CSV `song_lyrics.csv`)

---

## Step 1: Generate Song Lyrics (Baseline) -- DONE

Generate lyrics using `mistral-medium-latest` with minimal prompting -- just genre + theme.

- 40 songs generated (10 per genre), all traced in Weave
- Output: `outputs/baseline.jsonl`

---

## Step 2: Fine-Tune and Generate

### 2a: Data Pipeline -- DONE

- Extracted zip, filtered to pop/country/rock (English, 4-200 lines), deduped
- 1.84M train / 205K val songs
- Formatted as conversational JSONL in `data/sft/`

### 2b: Fine-Tune on HF Jobs -- IN PROGRESS

- Upload SFT dataset to HuggingFace Hub
- Launch LoRA fine-tuning on `mistralai/Mistral-7B-Instruct-v0.3` via `run_uv_job()`
- r=16, alpha=32, 3 epochs, a10g-large

### 2c: Generate with Fine-Tuned Model

- Use same 40 prompts from Step 1
- Generate with the fine-tuned model, save in Weave

---

## Step 3: Prompt Engineering and Generate -- DONE

- Genre-specific system prompts, structural instructions, few-shot examples
- 40 songs generated, all traced in Weave
- Output: `outputs/prompt_engineered.jsonl`

---

## Step 4: Pairwise Human Evaluation (x > y)

**Approach**: Blind pairwise comparison. For each genre/theme pair, show lyrics from two approaches side-by-side (randomized left/right to prevent position bias). Human picks a winner or declares a tie. No absolute scoring.

**Comparisons** (3 approaches = 3 pairs):
- Baseline vs Prompt-Engineered
- Baseline vs Fine-Tuned
- Prompt-Engineered vs Fine-Tuned

**Dimensions** (human picks winner per dimension):
- Overall quality ("which would you rather listen to?")
- Creativity / originality
- Genre fit
- Singability / flow
- Emotional impact

**Implementation**: Web app (`website/`) — shareable Flask app for crowdsourced blind judging.

- **Run**: `cd website && python app.py` → http://localhost:8080
- **Judge UI** (`/`): Two-column lyrics display, keyboard controls (J = A wins, K = B wins, Space = Tie). Shows lines 11–30 of each song for quicker judging.
- **Leaderboard** (`/leaderboard`): Live ELO standings with W/L/T records.
- **Blind eval**: Approach names are stripped before reaching the browser. The server resolves which approaches were compared when recording a judgment.
- **Per-session progress**: Each browser gets a random session ID (localStorage). Already-judged pairs are skipped on reload.

**Data storage**:
- `website/pairs.json` — All 126 A/B pairings baked in as a static JSON file. Rebuilt from `outputs/pairs.jsonl` by running `python website/build.py`.
- `website/judgments.db` — SQLite database (WAL mode), created automatically on first app startup. Contains a single `judgments` table with columns: `id, pair_index, genre, theme, left_approach, right_approach, rating, session_id, timestamp`. Existing judgments from `outputs/judgments.jsonl` are auto-imported on first run.
- **To export/backup**: just copy `website/judgments.db`. To reset: delete it and restart the app.

**Deployment**: Dockerfile included for containerized hosting (`gunicorn` on port 8080). Or just run `python app.py` directly.

**Analysis**: ELO computed on-demand from all judgments in SQLite via `/api/results`. Standalone `elo.py` still works against `outputs/judgments.jsonl` for the original 20 local judgments.

---

## Execution Order

```
Step 1 (Baseline Generation)          -- DONE
Step 2a (Data Pipeline)               -- DONE
Step 3 (Prompt Engineering)           -- DONE
Step 2b (Fine-Tune on HF Jobs)       -- IN PROGRESS
Step 2c (Fine-Tuned Generation)       -- after 2b
Step 4 (Pairwise Evaluation)          -- after all 3 sets exist
```

---

## Future (Milestone 7)

Recursive self-improvement loop with Claude + W&B MCP. Defer until Steps 1-4 complete.
