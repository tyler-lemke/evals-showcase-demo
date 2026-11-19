"""Put promptfoo's per-model results and the Jev batch runs side by side.

Reads results/promptfoo.json (npx promptfoo eval) and every results/jev_batch_g<size>.json
(batch_jev.py --group-size <size>) and prints one table: accuracy, latency and cost for each,
then how much of Jev's bill is fixed per-request overhead.

    uv run python compare.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PROMPTFOO = HERE / "results" / "promptfoo.json"


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def promptfoo_rows() -> tuple[list[dict], float]:
    data = json.loads(PROMPTFOO.read_text(encoding="utf-8"))
    by_model = defaultdict(list)
    for r in data["results"]["results"]:
        by_model[r["provider"]["id"].removeprefix("openrouter:")].append(r)
    rows = []
    for model, rs in by_model.items():
        n = len(rs)
        latencies = [r.get("latencyMs") or 0 for r in rs]
        rows.append({
            "name": model,
            "mode": "chat, 1 sentence per call",
            "n": n,
            "accuracy": sum(bool(r["success"]) for r in rs) / n,
            "mean_ms": statistics.mean(latencies),
            "p95_ms": p95(latencies),
            "wall_s": None,  # promptfoo doesn't record wall-clock per model
            "cost": sum(r.get("cost") or 0 for r in rs),
        })
    return rows, data["results"]["stats"]["durationMs"] / 1000


def jev_runs() -> list[dict]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in (HERE / "results").glob("jev_batch_g*.json")]
    return sorted(runs, key=lambda j: j["group_size"])


def jev_row(j: dict) -> dict:
    size = j["group_size"]
    return {
        "name": f"{j['model_requested']} (Jev)",
        "mode": f"decisions, {size} sentence{'s' if size > 1 else ''} per call",
        "n": j["n"],
        "accuracy": j["accuracy"],
        "mean_ms": j["latency_mean_s"] * 1000,
        "p95_ms": j["latency_p95_s"] * 1000,
        "wall_s": j["wall_clock_s"],
        "cost": j["cost_usd"],
    }


def print_table(header: list[str], table: list[list[str]]) -> None:
    table = [header] + table
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for i, row in enumerate(table):
        print("  ".join(c.ljust(widths[j]) if j < 2 else c.rjust(widths[j]) for j, c in enumerate(row)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main() -> None:
    rows, promptfoo_total_s = promptfoo_rows()
    runs = jev_runs()
    if not runs:
        print("No results/jev_batch_g*.json yet. Run `uv run python batch_jev.py` to add Jev.\n")
    rows += [jev_row(j) for j in runs]
    rows.sort(key=lambda r: (-r["accuracy"], r["cost"]))

    print_table(
        ["Model", "Mode", "N", "Accuracy", "Mean call", "p95 call", "Batch wall-clock", "Total cost", "$/1k sentences"],
        [
            [
                r["name"], r["mode"], str(r["n"]), f"{r['accuracy']:.1%}",
                f"{r['mean_ms']:.0f} ms", f"{r['p95_ms']:.0f} ms",
                f"{r['wall_s']:.1f}s" if r["wall_s"] is not None else "-",
                f"${r['cost']:.5f}", f"${r['cost'] / r['n'] * 1000:.4f}",
            ]
            for r in rows
        ],
    )

    if runs:
        o = runs[0]["overhead"]
        print(f"\nJev fixed overhead, measured on the real API: {o['overhead_tokens']} tokens (${o['overhead_cost_usd']:.6f}) "
              f"on every request, plus {o['question_tokens']} tokens (${o['question_cost_usd']:.6f}) per question.")
        print_table(
            ["Sentences per call", "Requests", "Billed", "Overhead part", "Without overhead", "$/1k billed", "$/1k without"],
            [
                [
                    str(j["group_size"]), str(j["requests"]), f"${j['cost_usd']:.6f}",
                    f"${j['overhead_total_usd']:.6f} ({j['overhead_total_usd'] / j['cost_usd']:.0%})",
                    f"${j['cost_without_overhead_usd']:.6f}",
                    f"${j['cost_usd'] / j['n'] * 1000:.4f}", f"${j['cost_without_overhead_usd'] / j['n'] * 1000:.4f}",
                ]
                for j in runs
            ],
        )

    ns = {r["n"] for r in rows}
    if len(ns) > 1:
        print(f"\nNote: sentence counts differ ({sorted(ns)}), so accuracy isn't like-for-like.")
    print("\nMean/p95 call = time of one API call. Batch wall-clock = start to finish of the whole batch; promptfoo")
    print(f"doesn't record it per model (its whole {sum(r['n'] for r in rows if r['wall_s'] is None)}-call eval, all models together, took {promptfoo_total_s:.0f}s).")


if __name__ == "__main__":
    main()
