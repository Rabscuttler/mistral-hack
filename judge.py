"""Pairwise human evaluation for song lyrics (x > y).

Keyboard-driven blind A/B comparison.
  j = left wins, k = right wins, space = tie
Cycles through dimensions, auto-submits, auto-advances.

Run with: streamlit run judge.py
"""

import json
import random
import time
from pathlib import Path

import streamlit as st

OUTPUTS_DIR = Path("outputs")
RESULTS_FILE = Path("outputs/judgments.jsonl")

DIMENSIONS = [
    ("overall", "Overall", "Which is better?"),
]


def load_outputs(approach: str) -> list[dict]:
    path = OUTPUTS_DIR / f"{approach}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_judgment(judgment: dict):
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(judgment) + "\n")


def load_judgments() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE) as f:
        return [json.loads(line) for line in f]


def main():
    st.set_page_config(page_title="Lyrics Judge", layout="wide")

    # Hide default Streamlit padding to maximize space
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0; }
        header[data-testid="stHeader"] { display: none; }
        .lyrics-box {
            height: 45vh;
            overflow-y: auto;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
            background: #ffffff;
            color: #111111;
            line-height: 1.4;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load approaches
    approaches = {}
    for approach in ["baseline", "prompt_engineered", "finetuned"]:
        data = load_outputs(approach)
        if data:
            approaches[approach] = data

    if len(approaches) < 2:
        st.warning(
            f"Need at least 2 approaches. Found: {list(approaches.keys())}. "
            f"Run generate.py first."
        )
        st.stop()

    # Sidebar config
    st.sidebar.header("Comparison")
    available = list(approaches.keys())
    approach_a = st.sidebar.selectbox("Approach A", available, index=0)
    remaining = [a for a in available if a != approach_a]
    approach_b = st.sidebar.selectbox("Approach B", remaining, index=0)

    # Build matching pairs
    index_a = {(r["genre"], r["theme"]): r for r in approaches[approach_a]}
    index_b = {(r["genre"], r["theme"]): r for r in approaches[approach_b]}
    common_keys = sorted(set(index_a.keys()) & set(index_b.keys()))

    if not common_keys:
        st.error("No matching genre/theme pairs.")
        st.stop()

    # Track already judged
    judgments = load_judgments()
    judged_keys = set()
    for j in judgments:
        pair = frozenset([j["left_approach"], j["right_approach"]])
        if pair == frozenset([approach_a, approach_b]):
            judged_keys.add((j["genre"], j["theme"]))

    # Init session state
    if "pair_idx" not in st.session_state:
        st.session_state.pair_idx = 0
    if "blind_seed" not in st.session_state:
        st.session_state.blind_seed = random.randint(0, 10000)

    idx = st.session_state.pair_idx
    idx = min(idx, len(common_keys) - 1)
    genre, theme = common_keys[idx]

    # Blind randomization
    rng = random.Random(st.session_state.blind_seed + idx)
    swapped = rng.random() > 0.5

    lyrics_a = index_a[(genre, theme)]
    lyrics_b = index_b[(genre, theme)]

    if swapped:
        left, right = lyrics_b, lyrics_a
        left_approach, right_approach = approach_b, approach_a
    else:
        left, right = lyrics_a, lyrics_b
        left_approach, right_approach = approach_a, approach_b

    # Handle query param submission from JS
    params = st.query_params
    if "ratings" in params:
        try:
            ratings_data = json.loads(params["ratings"])
            judgment = {
                "genre": genre,
                "theme": theme,
                "left_approach": left_approach,
                "right_approach": right_approach,
                "ratings": ratings_data,
                "timestamp": time.time(),
            }
            save_judgment(judgment)
            if idx < len(common_keys) - 1:
                st.session_state.pair_idx = idx + 1
            st.query_params.clear()
            st.rerun()
        except (json.JSONDecodeError, KeyError):
            st.query_params.clear()

    # Header line
    done = len(judged_keys)
    total = len(common_keys)
    already = " (done)" if (genre, theme) in judged_keys else ""
    st.markdown(
        f"**{idx + 1}/{total}{already}** &nbsp; `{genre}` — *{theme}*"
        f" &nbsp; [{done}/{total} judged]"
    )

    # Side-by-side lyrics in scrollable boxes
    import html as html_lib
    left_escaped = html_lib.escape(left["lyrics"])
    right_escaped = html_lib.escape(right["lyrics"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**J — Left**")
        st.markdown(f'<div class="lyrics-box">{left_escaped}</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown("**K — Right**")
        st.markdown(f'<div class="lyrics-box">{right_escaped}</div>', unsafe_allow_html=True)

    # Keyboard hint + JS injected directly into Streamlit page (not iframe)
    st.markdown(
        '<div style="text-align:center;padding:8px;color:#888;font-size:15px;">'
        '<kbd style="background:#333;border:1px solid #555;border-radius:4px;padding:2px 8px;font-family:monospace;">J</kbd> Left wins &nbsp;&nbsp;&nbsp;'
        '<kbd style="background:#333;border:1px solid #555;border-radius:4px;padding:2px 8px;font-family:monospace;">K</kbd> Right wins &nbsp;&nbsp;&nbsp;'
        '<kbd style="background:#333;border:1px solid #555;border-radius:4px;padding:2px 8px;font-family:monospace;">Space</kbd> Tie'
        '</div>',
        unsafe_allow_html=True,
    )
    # Inject keyboard listener directly into the main Streamlit document
    st.markdown("""
    <script>
    (function() {
        // Prevent duplicate listeners on rerun
        if (window._judgeKeyListenerAttached) return;
        window._judgeKeyListenerAttached = true;

        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

            let value = null;
            if (e.key === 'j' || e.key === 'J') { value = 'A wins'; }
            else if (e.key === 'k' || e.key === 'K') { value = 'B wins'; }
            else if (e.key === ' ') { value = 'Tie'; }

            if (value) {
                e.preventDefault();
                const ratings = JSON.stringify({"overall": value});
                const url = new URL(window.location.href);
                url.searchParams.set('ratings', ratings);
                window.location.href = url.toString();
            }
        });
    })();
    </script>
    """, unsafe_allow_html=True)

    # Sidebar: results
    st.sidebar.markdown("---")
    st.sidebar.subheader("Win Rates")
    all_judgments = load_judgments()
    if all_judgments:
        wins = {}
        ties = 0
        total = 0
        for j in all_judgments:
            rating = j["ratings"].get("overall", "Tie")
            total += 1
            if rating == "A wins":
                winner = j["left_approach"]
            elif rating == "B wins":
                winner = j["right_approach"]
            else:
                ties += 1
                continue
            wins[winner] = wins.get(winner, 0) + 1

        for app in sorted(wins.keys()):
            w = wins[app]
            st.sidebar.markdown(f"**{app}**: {w}/{total} ({100*w/total:.0f}%)")
        if ties:
            st.sidebar.markdown(f"Ties: {ties}/{total}")
    else:
        st.sidebar.info("No judgments yet.")

    if st.sidebar.checkbox("Reveal (after judging)"):
        st.sidebar.write(f"Left = **{left_approach}**")
        st.sidebar.write(f"Right = **{right_approach}**")


if __name__ == "__main__":
    main()
