"""SST-2 sentiment eval with DeepEval.

Every row of data/sst2_validation.csv becomes one pytest test: send the sentence to
the model through OpenRouter, then check that its reply exactly matches the label.
The pass rate DeepEval prints at the end is the model's accuracy.
"""

import csv
import os
import re
import time
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
MODEL = "anthropic/claude-haiku-5.5"  # any model id from https://openrouter.ai/models
NUM_EXAMPLES = 20  # first N rows, same as promptfoo's --filter-first-n 20. None = all 872.
PROMPT = (HERE / "prompt.txt").read_text(encoding="utf-8")

load_dotenv(HERE / ".env")
if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")

# OpenRouter speaks the OpenAI API, so the regular OpenAI client works with a different base URL.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def load_examples() -> list[dict]:
    with open(HERE / "data" / "sst2_validation.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[:NUM_EXAMPLES]


def classify(sentence: str) -> tuple[str, float]:
    """Return the model's reply and how many seconds the API call took."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.replace("{{sentence}}", sentence)}],
        temperature=0,  # promptfoo sends this by default; models that don't support it ignore it
    )
    seconds = time.perf_counter() - start
    reply = response.choices[0].message.content or ""
    # Lowercase and drop everything but letters: "Positive." -> "positive".
    # promptfooconfig.yaml normalizes replies the same way.
    return re.sub(r"[^a-z]", "", reply.lower()), seconds


@pytest.mark.parametrize("example", load_examples(), ids=lambda example: f"idx-{example['idx']}")
def test_sentiment(example: dict) -> None:
    answer, seconds = classify(example["sentence"])
    test_case = LLMTestCase(
        input=example["sentence"],
        actual_output=answer,
        expected_output=example["label"],
        completion_time=seconds,  # saved with the results, so reports can chart latency
    )
    assert_test(test_case, [ExactMatchMetric()])
