"""Custom Weave Scorers for song lyrics evaluation.

Scores lyrics by closeness to real song distributions (10K songs from Genius),
not by maximizing any single metric. A song that's too wordy, too varied,
or too rhyme-heavy is penalized just like one that's too short or repetitive.
"""

import json
import math
import os
import re
import time
from collections import Counter
from typing import ClassVar

import weave
from weave import Scorer


def _mistral_chat_with_retry(client, **kwargs) -> object:
    """Call Mistral chat.complete with exponential backoff on rate limits."""
    for attempt in range(5):
        try:
            return client.chat.complete(**kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** attempt + 1
                print(f"  [Rate limited] Retrying in {wait}s (attempt {attempt + 1}/5)")
                time.sleep(wait)
            else:
                raise
    return client.chat.complete(**kwargs)  # final attempt, let it raise


# Reference distributions from 10K real songs (data/processed/val.parquet)
REFERENCE = {
    "word_count": {"mean": 221, "std": 117},
    "unique_ratio": {"mean": 0.468, "std": 0.144},
    "line_count": {"mean": 34, "std": 17},
    "repeat_frac": {"mean": 0.138, "std": 0.092},
    "contraction_rate": {"mean": 0.050, "std": 0.041},
    "rhyme_rate": {"mean": 0.162, "std": 0.161},
}


def _gaussian_score(value: float, mean: float, std: float) -> float:
    """Score 0-1 based on closeness to reference distribution center."""
    if std == 0:
        return 1.0 if value == mean else 0.0
    z = (value - mean) / std
    return math.exp(-0.5 * z * z)


def _get_content_lines(lyrics: str) -> list[str]:
    """Extract non-empty, non-header lines from lyrics."""
    return [l.strip() for l in lyrics.split("\n")
            if l.strip() and not l.strip().startswith("[")]


def _rhyme_suffix(word: str, n: int = 2) -> str:
    """Get the last n characters of a word for rhyme matching."""
    cleaned = word.strip(".,!?;:\"'()-")
    return cleaned[-n:].lower() if len(cleaned) >= n else cleaned.lower()


class NaturalnessScorer(Scorer):
    """Scores lyrics by closeness to real song statistical distributions."""

    @weave.op()
    def score(self, output: dict) -> dict:
        lyrics = output["lyrics"]
        words = lyrics.lower().split()
        lines = _get_content_lines(lyrics)

        word_count = len(words)
        unique_ratio = len(set(words)) / max(word_count, 1)
        line_count = len(lines)

        # Repeat fraction: fraction of unique lines that appear more than once
        line_counter = Counter(lines)
        repeated = sum(1 for _l, c in line_counter.items() if c > 1)
        repeat_frac = repeated / max(len(line_counter), 1)

        # Contraction rate
        contraction_count = sum(1 for w in words if "'" in w and len(w) > 2)
        contraction_rate = contraction_count / max(word_count, 1)

        # Score each metric
        scores = {
            "word_count_score": _gaussian_score(word_count, **REFERENCE["word_count"]),
            "unique_ratio_score": _gaussian_score(unique_ratio, **REFERENCE["unique_ratio"]),
            "line_count_score": _gaussian_score(line_count, **REFERENCE["line_count"]),
            "repeat_frac_score": _gaussian_score(repeat_frac, **REFERENCE["repeat_frac"]),
            "contraction_score": _gaussian_score(contraction_rate, **REFERENCE["contraction_rate"]),
        }

        naturalness_score = sum(scores.values()) / len(scores)

        return {
            "naturalness_score": naturalness_score,
            "word_count": word_count,
            "unique_ratio": round(unique_ratio, 3),
            "line_count": line_count,
            "repeat_frac": round(repeat_frac, 3),
            "contraction_rate": round(contraction_rate, 4),
            **{k: round(v, 3) for k, v in scores.items()},
        }


class RhymeScorer(Scorer):
    """Scores rhyme density by closeness to real song rhyme patterns."""

    @weave.op()
    def score(self, output: dict) -> dict:
        lyrics = output["lyrics"]
        lines = _get_content_lines(lyrics)

        # Get end-words
        end_words = []
        for line in lines:
            tokens = line.split()
            if tokens:
                end_words.append(tokens[-1])

        # Check adjacent pairs for rhyme (last 2 chars match)
        rhymes = 0
        pairs = 0
        for i in range(0, len(end_words) - 1, 2):
            pairs += 1
            if _rhyme_suffix(end_words[i]) == _rhyme_suffix(end_words[i + 1]):
                rhymes += 1

        rhyme_rate = rhymes / max(pairs, 1)
        rhyme_score = _gaussian_score(rhyme_rate, **REFERENCE["rhyme_rate"])

        return {
            "rhyme_score": round(rhyme_score, 3),
            "rhyme_rate": round(rhyme_rate, 3),
            "rhyme_pairs": rhymes,
            "total_pairs": pairs,
        }


class AuthenticityScorer(Scorer):
    """Detects common LLM-isms that make lyrics feel artificial."""

    LLM_TELLS: ClassVar[list[str]] = [
        r"this song (captures?|is about|explores?|conveys?)",
        r"(here'?s|here is) (a|my|the) (song|lyrics)",
        r"title:\s",
        r"note:\s",
        r"i hope (you|this)",
    ]

    FORMAL_WORDS: ClassVar[list[str]] = [
        "furthermore", "moreover", "nevertheless", "consequently",
        "therefore", "whereas", "wherein", "hereby", "notwithstanding",
        "pertaining", "facilitate", "utilize", "endeavor", "commence",
    ]

    @weave.op()
    def score(self, output: dict) -> dict:
        lyrics = output["lyrics"]
        lyrics_lower = lyrics.lower()
        lines = lyrics.split("\n")
        content_lines = _get_content_lines(lyrics)
        header_lines = [l for l in lines if l.strip().startswith("[")]

        # Check for LLM meta-commentary
        meta_count = sum(
            1 for pattern in self.LLM_TELLS
            if re.search(pattern, lyrics_lower)
        )

        # Check for overly formal language
        words = lyrics_lower.split()
        formal_count = sum(1 for w in words if w in self.FORMAL_WORDS)

        # Header-to-content ratio (too many headers = LLM-ish)
        header_ratio = len(header_lines) / max(len(content_lines), 1)
        header_penalty = max(0, header_ratio - 0.15)  # Penalize if >15% headers

        # Contraction check — real songs use contractions heavily
        expandable = ["i am", "you are", "we are", "they are", "he is", "she is",
                       "it is", "do not", "does not", "did not", "will not",
                       "can not", "cannot", "would not", "could not", "should not"]
        formal_expansion_count = sum(1 for phrase in expandable if phrase in lyrics_lower)

        # Combine into score (1.0 = authentic, 0.0 = very LLM-ish)
        penalties = (
            meta_count * 0.25
            + formal_count * 0.1
            + header_penalty * 0.5
            + formal_expansion_count * 0.05
        )
        authenticity_score = max(0.0, 1.0 - penalties)

        return {
            "authenticity_score": round(authenticity_score, 3),
            "meta_commentary": meta_count,
            "formal_words": formal_count,
            "formal_expansions": formal_expansion_count,
            "header_ratio": round(header_ratio, 3),
        }


class LLMJudgeScorer(Scorer):
    """Uses Mistral as an LLM judge to rate lyrics on multiple dimensions."""

    @weave.op()
    def score(self, output: dict) -> dict:
        from mistralai import Mistral

        lyrics = output["lyrics"]
        genre = output["genre"]
        theme = output["theme"]

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        prompt = f"""Rate the following {genre} song lyrics about "{theme}" on these dimensions.
Score each 1-10 (1=terrible, 10=masterpiece). Be critical and honest.

LYRICS:
{lyrics}

Respond in this exact JSON format, nothing else:
{{
    "emotional_impact": <1-10>,
    "singability": <1-10>,
    "originality": <1-10>,
    "genre_fit": <1-10>,
    "overall": <1-10>,
    "critique": "<2-3 sentences identifying specific weaknesses>"
}}"""

        response = _mistral_chat_with_retry(
            client,
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        text = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            ratings = json.loads(text)
        except json.JSONDecodeError:
            return {
                "emotional_impact": 5.0,
                "singability": 5.0,
                "originality": 5.0,
                "genre_fit": 5.0,
                "overall": 5.0,
                "critique": f"[Parse error] {text[:200]}",
            }

        return {
            "emotional_impact": float(ratings.get("emotional_impact", 5)),
            "singability": float(ratings.get("singability", 5)),
            "originality": float(ratings.get("originality", 5)),
            "genre_fit": float(ratings.get("genre_fit", 5)),
            "overall": float(ratings.get("overall", 5)),
            "critique": str(ratings.get("critique", "")),
        }


class GenreClassifierScorer(Scorer):
    """Blind genre classification — does the song read as its intended genre?"""

    @weave.op()
    def score(self, output: dict) -> dict:
        from mistralai import Mistral

        lyrics = output["lyrics"]
        intended_genre = output["genre"]

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        prompt = f"""Read these song lyrics and classify the genre.
Choose exactly one: pop, country, rock, indie

LYRICS:
{lyrics}

Respond with just the genre name, nothing else."""

        response = _mistral_chat_with_retry(
            client,
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10,
        )

        predicted = response.choices[0].message.content.strip().lower()
        # Clean up response
        for g in ["pop", "country", "rock", "indie"]:
            if g in predicted:
                predicted = g
                break

        genre_match = predicted == intended_genre

        return {
            "genre_match": genre_match,
            "predicted_genre": predicted,
            "intended_genre": intended_genre,
        }
