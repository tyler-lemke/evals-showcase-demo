"""Classify a big batch of SST-2 sentences with Jev through OpenRouter's Decisions API.

Jev (TypeSafe) is a decision model, not a chat model. It has its own endpoint
(POST /api/alpha/decisions). You send `state` (the content) plus typed questions, and it
returns a typed answer per question: here a `choice` of positive/negative with a probability
per option and a confidence. It writes no text, output tokens are free, and every response
reports its own cost in `usage.cost`.

One request can carry several sentences: they all go in `state`, with one `choice` question
per sentence that points at it ("...in sentences.7..."). That matters for cost, because every
request carries a fixed amount of overhead tokens (Jev's own wrapping of state and questions)
on top of the sentence and the question. Bigger groups pay that overhead less often.

  --group-size 1    one sentence per request (one decision each, like the promptfoo evals)
  --group-size 100  100 sentences in a single request

Each run measures:
  - accuracy against the SST-2 labels (the same labels the promptfoo eval is graded on)
  - latency: wall-clock for the whole batch, and the time of each individual request
  - total cost, summed from every response's usage.cost
  - the fixed per-request overhead, measured on the real API with two empty requests (one
    question, then two), and what the cost would be without it

Usage:
    uv run python batch_jev.py                     # first 100 sentences, one per request
    uv run python batch_jev.py --group-size 25     # 25 sentences per request
    uv run python batch_jev.py --n 872 --group-size 50
Each group size saves results/jev_batch_g<size>.json; compare.py shows them next to promptfoo.
"""

import argparse
import csv
import json
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"  # or "~typesafe/jev-latest" to follow the newest release

_pause_until = 0.0  # all workers back off together when one gets rate limited
_lock = threading.Lock()


def load_examples(n: int | None) -> list[dict]:
    with open(HERE / "data" / "sst2_validation.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:n] if n else rows


def build_body(model: str, group: list[dict]) -> dict:
    """One request: every sentence in `state`, one choice question per sentence.

    Question ids are the sentence idx, which comes back as the key in `answers`. The model
    never sees the ids, so each question names its sentence's place in `state` instead."""
    return {
        "model": model,
        "state": {"sentences": {e["idx"]: e["sentence"] for e in group}},
        "questions": {
            e["idx"]: {
                "type": "choice",
                "instructions": f"Is the sentiment of the movie review sentence in sentences.{e['idx']} positive or negative?",
                "criteria": {"positive": None, "negative": None},
            }
            for e in group
        },
    }


def post(api_key: str, body: dict) -> tuple[int, dict, str]:
    """POST to the Decisions endpoint. Returns (status, response headers, response text)."""
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.status, dict(res.headers), res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def request_with_retry(api_key: str, body: dict, attempts: int = 5) -> tuple[dict, float]:
    """Send one Decisions request, retrying on 429 / 5xx / dropped connections.

    Returns (response json, seconds). The seconds are those of the request that finally
    succeeded; retries and the shared back-off wait count in the batch wall-clock only."""
    global _pause_until
    for attempt in range(attempts):
        wait = _pause_until - time.time()
        if wait > 0:
            time.sleep(wait)
        start = time.perf_counter()
        try:
            status, headers, text = post(api_key, body)
            seconds = time.perf_counter() - start
            if status == 200:
                answer = json.loads(text)
                if (answer.get("usage") or {}).get("cost") is None:  # never count a missing cost as free
                    raise RuntimeError("response had no usage.cost")
                return answer, seconds
            transient = status == 429 or status >= 500
            retry_after = headers.get("Retry-After", "")
            error = f"{status}: {text[:300]}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            transient, retry_after, error = True, "", f"transport error: {e}"
        if not transient or attempt + 1 == attempts:
            raise RuntimeError(error)
        backoff = float(retry_after) if retry_after.isdigit() else min(30, 2**attempt) + random.random()
        with _lock:
            _pause_until = max(_pause_until, time.time() + backoff)
    raise AssertionError("unreachable")


def classify_group(api_key: str, model: str, group: list[dict]) -> dict:
    answer, seconds = request_with_retry(api_key, build_body(model, group))
    usage = answer["usage"]
    sentences = []
    for e in group:
        choice = answer["answers"].get(e["idx"])  # None if the model left a question out
        predicted = choice["choice"] if choice else None
        sentences.append({
            "idx": e["idx"],
            "label": e["label"],
            "predicted": predicted,
            "correct": predicted == e["label"],
            "confidence": choice.get("confidence") if choice else None,
            "probabilities": choice.get("probabilities") if choice else None,
        })
    return {
        "latency_s": seconds,
        "input_tokens": usage.get("input_tokens"),
        "cost_usd": usage["cost"],
        "model_used": answer.get("model"),
        "sentences": sentences,
    }


def measure_overhead(api_key: str, model: str) -> dict:
    """Split a request's cost into fixed overhead and per-question cost, using the real API.

    Two requests with empty sentences, one question and then two. Their difference is what
    one extra question costs; whatever is left of the one-question request is overhead that
    every request pays, however many sentences it carries. Empty sentences add no tokens."""
    one = request_with_retry(api_key, build_body(model, [{"idx": "0", "sentence": ""}]))[0]["usage"]
    two = request_with_retry(api_key, build_body(model, [{"idx": "0", "sentence": ""}, {"idx": "1", "sentence": ""}]))[0]["usage"]
    question_tokens = two["input_tokens"] - one["input_tokens"]
    question_cost = two["cost"] - one["cost"]
    return {
        "overhead_tokens": one["input_tokens"] - question_tokens,
        "overhead_cost_usd": one["cost"] - question_cost,
        "question_tokens": question_tokens,
        "question_cost_usd": question_cost,
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="first N sentences (default 100, matches promptfoo's --filter-first-n 100)")
    parser.add_argument("--group-size", type=int, default=1, help="sentences per request (default 1)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests (default 8)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
    api_key = os.environ["OPENROUTER_API_KEY"]

    examples = load_examples(args.n)
    groups = [examples[i : i + args.group_size] for i in range(0, len(examples), args.group_size)]
    print(f"Classifying {len(examples)} sentences with {args.model}: {len(groups)} requests of up to "
          f"{args.group_size} sentences, {args.workers} at a time...")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        requests = list(pool.map(lambda g: classify_group(api_key, args.model, g), groups))  # keeps input order
    wall = time.perf_counter() - start

    overhead = measure_overhead(api_key, args.model)

    results = [s for r in requests for s in r["sentences"]]
    n = len(results)
    correct = sum(s["correct"] for s in results)
    missing = sum(s["predicted"] is None for s in results)
    latencies = [r["latency_s"] for r in requests]
    cost = sum(r["cost_usd"] for r in requests)
    overhead_total = len(requests) * overhead["overhead_cost_usd"]
    summary = {
        "model_requested": args.model,
        "model_used": sorted({r["model_used"] for r in requests}),
        "n": n,
        "group_size": args.group_size,
        "requests": len(requests),
        "workers": args.workers,
        "correct": correct,
        "missing": missing,
        "accuracy": correct / n,
        "wall_clock_s": wall,
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "input_tokens": sum(r["input_tokens"] or 0 for r in requests),
        "cost_usd": cost,
        "overhead": overhead,
        "overhead_total_usd": overhead_total,
        "cost_without_overhead_usd": cost - overhead_total,
        "results": results,
    }
    out = HERE / "results" / f"jev_batch_g{args.group_size}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Model:       {', '.join(summary['model_used'])}")
    print(f"Accuracy:    {correct}/{n} = {summary['accuracy']:.1%}" + (f"  ({missing} sentences got no answer)" if missing else ""))
    print(f"Latency:     {wall:.1f}s wall-clock for the batch ({n / wall:.1f} sentences/s); "
          f"per request mean {summary['latency_mean_s'] * 1000:.0f} ms, p50 {summary['latency_p50_s'] * 1000:.0f} ms, "
          f"p95 {summary['latency_p95_s'] * 1000:.0f} ms")
    print(f"Cost:        ${cost:.6f} total (${cost / n * 1000:.4f} per 1,000 sentences), "
          f"{summary['input_tokens']} input tokens ({summary['input_tokens'] / n:.0f} per sentence), output tokens are free")
    print(f"Overhead:    {overhead['overhead_tokens']} tokens (${overhead['overhead_cost_usd']:.6f}) per request, measured on the real API; "
          f"{len(requests)} requests -> ${overhead_total:.6f} of the bill")
    print(f"Without it:  ${summary['cost_without_overhead_usd']:.6f} (${summary['cost_without_overhead_usd'] / n * 1000:.4f} per 1,000 sentences)")
    wrong = [s for s in results if not s["correct"]]
    if wrong:
        print("Misses:      " + ", ".join(f"idx {s['idx']} (said {s['predicted']}, confidence {s['confidence']})" for s in wrong))
    print(f"Saved to {out.relative_to(HERE)}. Run compare.py to see it next to promptfoo.")


if __name__ == "__main__":
    main()
