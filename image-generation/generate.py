"""Generate one image per prompt per model, in parallel, and save the results.

This is the only script that calls an image model. promptfoo and DeepEval both grade the
files it writes (results/images/ and results/images.csv), so both tools judge the exact
same images.

Images that already exist are kept, so running it again only makes what's missing (a new
prompt or a new model) and you pay for each image once. Delete results/images/ to start over.

Run from the image-generation folder:  uv run python generate.py
"""

import base64
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
IMAGES_DIR = HERE / "results" / "images"
OUTPUT_FILE = HERE / "results" / "images.csv"
FIELDS = ["prompt_id", "prompt", "model", "image", "mime", "latency_s", "cost_usd", "error"]

with (HERE / "prompts.csv").open(newline="", encoding="utf-8") as f:
    PROMPTS = list(csv.DictReader(f))  # columns: id, prompt

# Image models on OpenRouter: https://openrouter.ai/models?output_modalities=image
MODELS = [
    "openai/gpt-image-2",  # OpenAI's image model, known for clean text
    "google/gemini-3-pro-image",  # Google's flagship ("Nano Banana Pro")
    "black-forest-labs/flux.2-pro",  # the most popular open-weight family
    "google/gemini-2.5-flash-image",  # an older model that often garbles text in images
]

load_dotenv(HERE / ".env")
if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def request_image(model: str, prompt: str) -> tuple[str, str, object]:
    """Returns (mime type, base64 data, usage). Most OpenRouter image models work through chat
    completions; a few (such as gpt-image-2) only through the images endpoint, so fall back to it."""
    error = None
    for modalities in (["image", "text"], ["image"]):  # some models also return text, some don't
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"modalities": modalities},
            )
        except Exception as e:
            error = e
            continue
        images = (response.choices[0].message.model_extra or {}).get("images") or []
        if not images:
            raise RuntimeError(f"no image returned: {response.choices[0].message.content!r}")
        header, data = images[0]["image_url"]["url"].split(",", 1)  # data:image/png;base64,...
        return header.removeprefix("data:").split(";")[0], data, response.usage
    if "images endpoint" not in str(error):
        raise error
    response = client.images.generate(model=model, prompt=prompt)
    return "image/png", response.data[0].b64_json, response.usage


def generate(prompt: dict, model: str) -> dict:
    """Ask one model for one prompt's image. Returns one row of results/images.csv."""
    label = f"{prompt['id']} / {model}"
    row = {"prompt_id": prompt["id"], "prompt": prompt["prompt"], "model": model}
    start = time.perf_counter()
    try:
        mime, data, usage = request_image(model, prompt["prompt"])
    except Exception as error:  # keep going so one failing model doesn't lose the others
        print(f"  {label} FAILED: {error}")
        return {**row, "image": "", "mime": "", "latency_s": "", "cost_usd": "", "error": str(error)}
    seconds = time.perf_counter() - start

    path = IMAGES_DIR / f"{prompt['id']}__{model.replace('/', '__')}.{EXTENSIONS.get(mime, 'png')}"
    path.write_bytes(base64.b64decode(data))
    cost = getattr(usage, "cost", None) if usage else None  # OpenRouter adds the price in USD
    print(f"  {label}: {seconds:.1f} s, ${cost}, saved {path.name}")
    return {
        **row,
        "image": path.relative_to(HERE).as_posix(),
        "mime": mime,
        "latency_s": round(seconds, 2),
        "cost_usd": cost,
        "error": "",
    }


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    # Keep rows from earlier runs whose image is still on disk.
    kept = {}
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("image") and (HERE / row["image"]).exists():
                    kept[(row["prompt_id"], row["model"])] = row

    jobs = [(p, m) for p in PROMPTS for m in MODELS if (p["id"], m) not in kept]
    print(f"{len(kept)} images already made, generating {len(jobs)} in parallel ...")
    with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
        made = {(r["prompt_id"], r["model"]): r for r in pool.map(lambda job: generate(*job), jobs)}

    rows = [kept.get((p["id"], m)) or made[(p["id"], m)] for p in PROMPTS for m in MODELS]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
