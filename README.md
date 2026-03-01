# Mistral Global Hack 

Better song lyrics with supervised fine tuning with Mistral-7B.
AI-generated song lyrics — baseline vs prompt-engineered vs fine-tuned Mistral models, evaluated by automated scorers and blind human judgment.

[WandB report](https://wandb.ai/furbnow/mistral-hackathon/reports/Mistral-Global-Hackathon-Improving-GenAI-Song-Lyrics--VmlldzoxNjA2OTg3NQ?accessToken=q0wdn6shg3inryd3z136yfl0d9ig42332xwsg4aigck675fkvrpo2e4ipt0wx6yb)

BONUS SONGS:    
[ElevenLabs - Ship it broken, Ship it Wild!](https://elevenlabs.io/music/songs/qA6JijrpX09dmHf8emTb) - skip to 48s for the bangin' chorus  
[Ship it before sunrise (Suno)](https://suno.com/s/3WCo7L9cFqXLj9Nc)

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
