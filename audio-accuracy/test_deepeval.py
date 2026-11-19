"""Speech-to-text accuracy on the Contoso Pharmacy call, graded with DeepEval.

transcribe.py has already sent the audio to every model and saved the transcripts in
results/transcripts.csv. This file only grades them: one pytest test per model, with
three metrics run on the same transcript. promptfooconfig.yaml grades the same file.
"""

import csv
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric, GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv

from metrics import MAX_WER, MIN_KEY_TERM_RECALL, key_term_recall, word_error_rate

HERE = Path(__file__).parent
TRANSCRIPTS = HERE / "results" / "transcripts.csv"
AUDIO_FILE = "data/contoso_pharmacy_call.wav"
REFERENCE = (HERE / "data" / "contoso_pharmacy_reference.txt").read_text(encoding="utf-8")
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"  # the same judge promptfoo uses

load_dotenv(HERE / ".env")  # OPENROUTER_API_KEY, read by OpenRouterModel


class WordErrorRate(BaseMetric):
    """WER from metrics.py. Lower is better, so it passes at or below the threshold."""

    def __init__(self, threshold: float = MAX_WER):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        result = word_error_rate(test_case.expected_output, test_case.actual_output)
        self.score, self.reason, self.success = result["score"], result["reason"], result["pass"]
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "WER"


class KeyTermRecall(BaseMetric):
    """Share of drug, dose, date and name mentions the transcript got right, from metrics.py."""

    def __init__(self, threshold: float = MIN_KEY_TERM_RECALL):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        result = key_term_recall(test_case.expected_output, test_case.actual_output)
        self.score, self.reason, self.success = result["score"], result["reason"], result["pass"]
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Key terms"


def likert_metric() -> GEval:
    """A judge model rates meaning on the 1-5 scale in likert_scale.txt.

    GEval maps the rating onto 0-1 (1 -> 0, 4 -> 0.75, 5 -> 1), so a threshold of 0.75
    means "4 or better", the same as in promptfooconfig.yaml.
    """
    scale = (HERE / "likert_scale.txt").read_text(encoding="utf-8").splitlines()
    rubric = []
    for line in scale:
        rating, description = line.split(":", 1)
        rubric.append(Rubric(score_range=(int(rating), int(rating)), expected_outcome=description.strip()))
    return GEval(
        name="Likert",
        evaluation_steps=[
            "The actual output is a speech-to-text transcript of a phone call to a pharmacy. "
            "The expected output is the reference transcript of the same call.",
            "Compare them and rate how faithfully the actual output records what was said.",
            "Ignore fillers (uh, um), false starts, punctuation, capitalization, speaker line "
            "breaks, and how numbers are written (2021 or twenty twenty-one).",
        ],
        rubric=rubric,
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=OpenRouterModel(model=JUDGE_MODEL),
        threshold=0.75,  # pass at 4 or 5
    )


def load_transcripts() -> list[dict]:
    if not TRANSCRIPTS.exists():
        raise FileNotFoundError(f"{TRANSCRIPTS} not found. Run `uv run python transcribe.py` first.")
    with TRANSCRIPTS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", load_transcripts(), ids=lambda row: row["model"])
def test_transcript(row: dict) -> None:
    test_case = LLMTestCase(
        input=AUDIO_FILE,
        actual_output=row["transcript"],
        expected_output=REFERENCE,
        # Saved with DeepEval's results, so reports can chart speed and price next to accuracy.
        completion_time=float(row["latency_s"]) if row["latency_s"] else None,
        additional_metadata={"model": row["model"], "cost_usd": row["cost_usd"], "error": row["error"]},
    )
    assert_test(test_case, [WordErrorRate(), KeyTermRecall(), likert_metric()])
