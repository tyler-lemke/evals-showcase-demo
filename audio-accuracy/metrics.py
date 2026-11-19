"""Accuracy metrics shared by promptfoo and DeepEval, so both tools report the same numbers.

- Word error rate (WER): the share of reference words the transcript got wrong.
- Key-term recall: the share of mentions of the words in data/key_terms.txt that the
  transcript got right.

Each metric returns {"pass", "score", "reason"}, which is the format promptfoo expects from
a Python assertion. test_deepeval.py wraps the same functions as DeepEval metrics.

This file uses only the Python standard library, because promptfoo runs it with whatever
`python` is on your PATH, not with this folder's virtual environment.
"""

import re
from pathlib import Path

MAX_WER = 0.05  # pass if at most 1 word in 20 is wrong, about what careful human transcribers manage
MIN_KEY_TERM_RECALL = 1.0  # pass only if every mention of every key term is right

# Sounds that carry no meaning. Models disagree on whether to write them down,
# so they're removed from both the reference and the transcript before comparing.
FILLERS = {"uh", "um", "umm", "uhm", "er", "erm", "ah", "hmm", "mm", "mmm", "mhm"}

# Different ways of writing the same word.
SPELLINGS = {
    "okay": "ok",
    "ahold": "a hold",
    "callback": "call back",
    "cancelling": "canceling",
    "mg": "milligrams",
}

KEY_TERMS_FILE = Path(__file__).parent / "data" / "key_terms.txt"
KEY_TERMS = [
    line.split("|")
    for line in KEY_TERMS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation and fillers, unify spellings: "Okay, uh, 2021." -> ["ok", "2021"]."""
    text = text.lower().replace("’", "").replace("'", "")  # "you're" -> "youre"
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)  # "500mg" -> "500 mg"
    words = re.sub(r"[^a-z0-9]+", " ", text).split()  # punctuation and hyphens become spaces
    words = " ".join(SPELLINGS.get(word, word) for word in words).split()
    return [word for word in words if word not in FILLERS]


def count_errors(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Fewest (substitutions, deletions, insertions) that turn the reference into the hypothesis.

    This is the classic edit distance, counted in words instead of letters.
    Each cell holds (total errors, substitutions, deletions, insertions).
    """
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]  # empty reference: all insertions
    for i, reference_word in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]  # empty hypothesis: all deletions
        for j, hypothesis_word in enumerate(hypothesis, start=1):
            total, subs, dels, ins = previous[j - 1]
            if reference_word == hypothesis_word:
                same_or_swapped = (total, subs, dels, ins)
            else:
                same_or_swapped = (total + 1, subs + 1, dels, ins)
            total, subs, dels, ins = previous[j]
            deleted = (total + 1, subs, dels + 1, ins)
            total, subs, dels, ins = current[j - 1]
            inserted = (total + 1, subs, dels, ins + 1)
            current.append(min(same_or_swapped, deleted, inserted))
        previous = current
    _, subs, dels, ins = previous[-1]
    return subs, dels, ins


def word_error_rate(reference: str, hypothesis: str) -> dict:
    """WER = (wrong + missing + extra words) / words in the reference. Lower is better."""
    reference_words, hypothesis_words = normalize(reference), normalize(hypothesis)
    subs, dels, ins = count_errors(reference_words, hypothesis_words)
    wer = (subs + dels + ins) / len(reference_words)
    return {
        "pass": wer <= MAX_WER,
        "score": wer,
        "reason": f"WER {wer:.1%}: {subs} wrong, {dels} missing, {ins} extra "
        f"(reference has {len(reference_words)} words)",
    }


def count_mentions(words: list[str], spellings: list[str]) -> int:
    text = " ".join(words)
    phrases = {" ".join(normalize(spelling)) for spelling in spellings}
    return sum(len(re.findall(rf"(?<!\S){re.escape(phrase)}(?!\S)", text)) for phrase in phrases)


def key_term_recall(reference: str, hypothesis: str) -> dict:
    """Share of key-term mentions in the reference that the transcript also has. Higher is better."""
    reference_words, hypothesis_words = normalize(reference), normalize(hypothesis)
    said = heard = 0
    missed = []
    for spellings in KEY_TERMS:
        in_reference = count_mentions(reference_words, spellings)
        in_transcript = min(count_mentions(hypothesis_words, spellings), in_reference)
        said += in_reference
        heard += in_transcript
        if in_transcript < in_reference:
            missed.append(f"{spellings[0]} ({in_transcript} of {in_reference})")
    recall = heard / said
    reason = f"{heard} of {said} key-term mentions right"
    if missed:
        reason += f"; missed: {', '.join(missed)}"
    return {"pass": recall >= MIN_KEY_TERM_RECALL, "score": recall, "reason": reason}


# promptfoo calls these with the transcript and the test's variables (see promptfooconfig.yaml).


def promptfoo_wer(output: str, context: dict) -> dict:
    return word_error_rate(context["vars"]["reference"], output)


def promptfoo_key_terms(output: str, context: dict) -> dict:
    return key_term_recall(context["vars"]["reference"], output)
