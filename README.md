# Mistral Lyrics Arena

AI-generated song lyrics — baseline vs prompt-engineered vs fine-tuned Mistral models, evaluated by automated scorers and blind human judgment.

## Live Site

Browse and compare generated lyrics at the GitHub Pages site (configure Pages to serve from `docs/` on `main`).

## Structure

- `generate.py` — Generate lyrics (baseline, prompt-engineered, fine-tuned)
- `eval_loop.py` — Self-improving prompt optimization loop with Weave evaluations
- `scorers.py` — Naturalness, rhyme, authenticity, LLM-judge, genre classifiers
- `website/` — Flask app for human A/B evaluation + song browsing
- `website/freeze.py` — Frozen-Flask build for static GitHub Pages deployment
- `finetune/` — LoRA fine-tuning scripts for Mistral-7B via HuggingFace
- `outputs/` — Generated lyrics (JSONL) and pre-computed A/B pairs
- `artifacts/` — Prompt iterations and optimization results

## Static Site

Rebuild and deploy:

```bash
python3 website/freeze.py   # outputs to docs/
git add docs && git commit -m "rebuild static site"
git push
```

## Dev Server

```bash
cd website && python3 app.py
```
