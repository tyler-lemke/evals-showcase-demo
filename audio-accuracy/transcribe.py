"""Transcribe the Contoso Pharmacy call once with every model and save the results.

This is the only script that sends audio to a model. promptfoo and DeepEval both grade the
file it writes, results/transcripts.csv, so every metric in both tools scores the exact same
transcripts, and you pay for each model's transcription once per run.

Run from the audio-accuracy folder:  uv run python transcribe.py
"""

import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
AUDIO_FILE = HERE / "data" / "contoso_pharmacy_call.wav"
OUTPUT_FILE = HERE / "results" / "transcripts.csv"

# Speech-to-text models on OpenRouter: https://openrouter.ai/models?output_modalities=transcription
MODELS = [
    "openai/whisper-1",  # the classic baseline
    "openai/gpt-4o-mini-transcribe",  # OpenAI's low-cost newer model
    "deepgram/nova-3",  # widely used for call centers
    "elevenlabs/scribe-v2-medical",  # tuned for medical terms such as drug names
    "google/gemini-3.5-transcribe",  # Google's speech-to-text model
    "microsoft/mai-transcribe-2",  # Microsoft's current model (the original sample used Azure Speech)
]

load_dotenv(HERE / ".env")
if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")

# OpenRouter's /audio/transcriptions endpoint speaks the OpenAI API, so the OpenAI client works.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def transcribe(model: str, audio: bytes) -> dict:
    """Send the whole call to one model. Returns one row of results/transcripts.csv."""
    start = time.perf_counter()
    try:
        result = client.audio.transcriptions.create(
            model=model,
            file=(AUDIO_FILE.name, audio, "audio/wav"),
            language="en",
        )
    except Exception as error:  # keep going so one failing model doesn't lose the others
        print(f"  FAILED: {error}")
        return {"model": model, "transcript": "", "latency_s": "", "cost_usd": "", "error": str(error)}
    seconds = time.perf_counter() - start
    cost = result.usage.cost if result.usage else None  # OpenRouter adds the price in USD
    return {
        "model": model,
        "transcript": result.text.strip(),
        "latency_s": round(seconds, 2),
        "cost_usd": cost,
        "error": "",
    }


def main() -> None:
    audio = AUDIO_FILE.read_bytes()
    rows = []
    for model in MODELS:
        print(f"Transcribing with {model} ...")
        row = transcribe(model, audio)
        if not row["error"]:
            print(f"  {row['latency_s']} s, ${row['cost_usd']}, {len(row['transcript'].split())} words")
        rows.append(row)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "transcript", "latency_s", "cost_usd", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} transcripts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
