"""Image generation, graded with DeepEval.

generate.py has already asked every model for the image and saved it in results/images/.
This file only grades them: one pytest test per prompt per model. Each image goes into an MLLMImage,
and a vision judge rates it with GEval on the 1-5 scale in likert_scale.txt.
promptfooconfig.yaml grades the same images.
"""

import csv
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, MLLMImage, SingleTurnParams
from dotenv import load_dotenv

HERE = Path(__file__).parent
IMAGES_CSV = HERE / "results" / "images.csv"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"  # the same judge promptfoo uses; it can see images

load_dotenv(HERE / ".env")  # OPENROUTER_API_KEY, read by OpenRouterModel


def likert_metric() -> GEval:
    """GEval maps the 1-5 rating onto 0-1 (4 -> 0.75), so 0.75 means "4 or better"."""
    rubric = []
    for line in (HERE / "likert_scale.txt").read_text(encoding="utf-8").splitlines():
        rating, description = line.split(":", 1)
        rubric.append(Rubric(score_range=(int(rating), int(rating)), expected_outcome=description.strip()))
    return GEval(
        name="Likert",
        evaluation_steps=[
            "The input is the prompt given to an image generation model. The actual output is the image it made.",
            "Read any text in the image letter by letter, exactly as written, before comparing it with the prompt. "
            "Do not assume the text is correct because the prompt says what it should be.",
            "Rate how well the image matches the prompt on the rubric.",
        ],
        rubric=rubric,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=OpenRouterModel(model=JUDGE_MODEL),
        threshold=0.75,  # pass at 4 or 5
    )


def load_images() -> list[dict]:
    if not IMAGES_CSV.exists():
        raise FileNotFoundError(f"{IMAGES_CSV} not found. Run `uv run python generate.py` first.")
    with IMAGES_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", load_images(), ids=lambda row: f"{row['prompt_id']}-{row['model']}")
def test_image(row: dict) -> None:
    if row["error"]:
        pytest.fail(f"{row['model']} made no image: {row['error']}")
    image = MLLMImage(url=str(HERE / row["image"]))
    test_case = LLMTestCase(
        input=row["prompt"],
        actual_output=f"{image}",  # DeepEval swaps this placeholder for the image itself
        completion_time=float(row["latency_s"]) if row["latency_s"] else None,
        additional_metadata={"prompt_id": row["prompt_id"], "model": row["model"], "cost_usd": row["cost_usd"]},
    )
    assert_test(test_case, [likert_metric()])
