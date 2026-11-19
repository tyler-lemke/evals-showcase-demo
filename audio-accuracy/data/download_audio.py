"""Download the Contoso Pharmacy sample call and save a mono copy for transcription.

The call comes from Microsoft's call-center sample (MIT license):
https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/scenarios/call-center/sampledata

The original is stereo with one speaker per channel: the pharmacy agent on the left, the
customer on the right. This script mixes the two channels into one, so every model hears
the whole conversation the way a single-channel phone recording would sound.

Run from the audio-accuracy folder:  uv run python data/download_audio.py
"""

import array
import io
import wave
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = (
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/"
    "scenarios/call-center/sampledata/Call3_separated_16k_pharmacy_call.wav"
)
OUTPUT_FILE = Path(__file__).parent / "contoso_pharmacy_call.wav"
INT16_MIN, INT16_MAX = -32768, 32767


def download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def stereo_to_mono(wav_bytes: bytes) -> tuple[bytes, int]:
    """Add the left and right channels together. Returns (mono samples, sample rate)."""
    with wave.open(io.BytesIO(wav_bytes)) as source:
        if source.getnchannels() != 2 or source.getsampwidth() != 2:
            raise ValueError("Expected a 16-bit stereo WAV file")
        rate = source.getframerate()
        samples = array.array("h", source.readframes(source.getnframes()))
    # Samples alternate left, right, left, right. Only one person talks at a time, so adding
    # the channels keeps each voice at its original volume; clamp in case both overlap.
    left, right = samples[0::2], samples[1::2]
    mono = array.array("h", (max(INT16_MIN, min(INT16_MAX, a + b)) for a, b in zip(left, right)))
    return mono.tobytes(), rate


def main() -> None:
    mono, rate = stereo_to_mono(download(SOURCE_URL))
    with wave.open(str(OUTPUT_FILE), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(mono)
    print(f"Saved {len(mono) / 2 / rate:.1f} seconds of mono audio to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
