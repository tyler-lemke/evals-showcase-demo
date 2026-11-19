"""Download the SST-2 validation split from Hugging Face and save it as a CSV.

SST-2 (Stanford Sentiment Treebank) is a set of single sentences from movie reviews,
each labelled positive or negative. The validation split has 872 sentences.
Dataset page: https://huggingface.co/datasets/stanfordnlp/sst2

Run from the sst2-sentiment folder:  uv run python data/download_sst2.py
"""

import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

API_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # the API returns at most 100 rows per request
LABELS = {0: "negative", 1: "positive"}  # SST-2 stores labels as integers
OUTPUT_FILE = Path(__file__).parent / "sst2_validation.csv"


def fetch_page(offset: int) -> dict:
    query = urlencode(
        {
            "dataset": "stanfordnlp/sst2",
            "config": "default",
            "split": "validation",
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    with urlopen(f"{API_URL}?{query}", timeout=30) as response:
        return json.load(response)


def download_rows() -> list[dict]:
    rows = []
    while True:
        page = fetch_page(offset=len(rows))
        rows += [
            {
                "idx": item["row"]["idx"],
                "sentence": item["row"]["sentence"].strip(),
                "label": LABELS[item["row"]["label"]],
            }
            for item in page["rows"]
        ]
        if not page["rows"] or len(rows) >= page["num_rows_total"]:
            return rows


def main() -> None:
    rows = download_rows()
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "sentence", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
