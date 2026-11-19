# Speech-to-text accuracy evals

How accurately do speech-to-text models transcribe a customer service call? This folder sends
one recorded pharmacy call to six models through [OpenRouter](https://openrouter.ai), saves the
transcripts, then grades them twice: once with [promptfoo](https://promptfoo.dev) and once with
[DeepEval](https://deepeval.com).

## The audio

`data/contoso_pharmacy_call.wav` is a 2-minute call to Contoso Pharmacy. It's the same call that
opens Microsoft's video *Call center transcription and analysis using Azure AI*. A customer named
Mary wants a refill of her cholesterol medication. The agent finds that her doctor stopped the
generic Lipitor in 2021 and switched her to metformin 500 milligrams.

**Source:** `Call3_separated_16k_pharmacy_call.wav` from Microsoft's
[call-center sample](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/scenarios/call-center).
The original is stereo, with the agent on one channel and the customer on the other.
`data/download_audio.py` downloads it again and mixes it into one channel, the way an ordinary
phone recording sounds.

**License:** MIT, copyright Microsoft. See `data/LICENSE-audio.md`.

## The reference transcript

Word error rate needs a correct transcript to compare against, and the sample doesn't include one.
`data/contoso_pharmacy_reference.txt` was built by comparing nine speech-to-text models with the
transcript that Azure Speech made of the same file in 2022. They agree on almost every word. Where
they disagreed, the reference uses what most of them heard:

| Time | Reference (most models) | Some models heard |
|------|-------------------------|-------------------|
| 0:04 | calling **about a** refill | calling **to get a** refill |
| 0:22 | prescription drugs that my doctor | that **the doc-** my doctor (a false start) |
| 0:29 | Okay, certainly happy | Okay, **I'm** certainly happy |
| 0:36 | you **were** on a generic form | you **are** on |
| 0:56 | **I'd** be happy to check | be happy to check |
| 1:25 | Oh, **huh.** I mean | Oh, I mean |
| 1:32 | provided by **your** doctor's office | **a** doctor's office, doctor's office |

Listen to those spots if you want to be sure. If you hear something different, edit the file;
every grade is measured against it.

The reference is *clean verbatim*: it leaves out fillers (uh, um) and false starts, writes numbers
as digits, and puts each speaker turn on its own line, starting with the agent.

## What gets measured

| Metric | What it catches | How | Passes at |
|--------|-----------------|-----|-----------|
| **WER** (word error rate) | Every wrong, missing or extra word | `metrics.py` counts word edits against the reference | 5% or less |
| **Key terms** | Errors in the words a pharmacy can't get wrong: drug names, dose, dates and names (`data/key_terms.txt`) | `metrics.py` checks every mention | 100% |
| **Likert** (1 to 5) | Whether the meaning came through | A judge model compares the transcript with the reference using `likert_scale.txt` | 4 or 5 |

Each metric covers a blind spot in the others. WER treats every word the same, so "twenty
twenty-one" for "2021" costs three errors even though nothing is wrong, while "Contessa" for
"Contoso" costs just one. Key terms catch the errors that matter most. The Likert judge reads
for meaning the way a person would, but judges are less consistent than code.

Before comparing words, `metrics.py` lowercases both texts, removes punctuation and fillers, and
treats a few spellings as the same (`okay` and `ok`, `mg` and `milligrams`). It doesn't convert
numbers written as words, so a model that writes them out takes a small WER penalty.

Each model's speed and price are also saved (`latency_s` and `cost_usd` in
`results/transcripts.csv`), but they don't count toward pass or fail.

## One transcription run, two graders

```
data/contoso_pharmacy_call.wav
        │
        │  transcribe.py   the only step that sends audio to a model
        ▼
results/transcripts.csv    one row per model: transcript, latency, cost
        │
        ├──▶ promptfoo     promptfooconfig.yaml
        └──▶ DeepEval      test_deepeval.py
```

- The audio goes to each model once per run, and both tools grade the same saved transcripts.
- Inside each tool, all three metrics score the same transcript.
- WER and key terms are plain code in `metrics.py`. Both tools import it, so their numbers always
  match. Only the Likert rating calls a model.
- To change a threshold, a key term or the judge, run the graders again without transcribing again.

## Setup

You need [uv](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org), Python 3 on your PATH
(promptfoo uses it to run `metrics.py`) and an [OpenRouter API key](https://openrouter.ai/keys).

```bash
cd audio-accuracy
npm install            # promptfoo, plus a local Node 22 (promptfoo needs 22 or newer)
uv sync                # Python environment with DeepEval
cp .env.example .env   # then paste your OpenRouter key into .env
```

## Run

| Step | Command | What you get |
|------|---------|--------------|
| 1. Transcribe | `uv run python transcribe.py` | `results/transcripts.csv`. Takes about 20 seconds and costs about 5 cents |
| 2a. Grade with promptfoo | `npx promptfoo eval` | One row per model, with WER, Key terms and Likert scores |
| | `npx promptfoo view` | The same results in your browser |
| 2b. Grade with DeepEval | `uv run deepeval test run test_deepeval.py` | One pytest test per model, then the pass rate |

Each grading run costs about 4 cents for the judge. Running `transcribe.py` again replaces
`results/transcripts.csv`, so the graders always score the latest transcripts. As in
`sst2-sentiment`, promptfoo's cache is off and DeepEval writes a new `results/test_run_*.json`
each time, so one run's grades never mix with another's.

**Some tests will fail.** A failed test means a model got something wrong, which is what this
eval is meant to find. Both grading commands exit with an error code when any test fails, and
that's expected.

## Results from the first run (2026-11-18)

| Model | WER | Key terms | Likert (promptfoo / DeepEval) | Time | Cost |
|-------|-----|-----------|-------------------------------|------|------|
| openai/whisper-1 | 1.3% | 13 of 13 | 4 / 4 | 7.1 s | $0.012 |
| openai/gpt-4o-mini-transcribe | 1.3% | 13 of 13 | 5 / 5 | 3.7 s | $0.003 |
| deepgram/nova-3 | 4.7% | 13 of 13 | 5 / 5 | 0.8 s | $0.009 |
| elevenlabs/scribe-v2-medical | 1.3% | 13 of 13 | 5 / 4 | 2.1 s | $0.007 |
| google/gemini-3.5-transcribe | 0.7% | 12 of 13: "Contessa" | 3 / 3 | 4.4 s | $0.011 |
| microsoft/mai-transcribe-2 | 1.3% | 12 of 13: "Contosa" | 5 / 4 | 1.4 s | $0.003 |

Gemini had the lowest WER but misheard the pharmacy's name. Deepgram had the highest WER mostly
because it writes numbers as words, which the judge ignored. The two judges also disagreed:
"Contessa" got a 3 but "Contosa" got a 4 or 5. Run it again and the Likert column can change
by a point; WER and key terms won't.

## Change what's tested

| To change | Edit |
|-----------|------|
| The models | `MODELS` in `transcribe.py` ([speech-to-text models on OpenRouter](https://openrouter.ai/models?output_modalities=transcription)) |
| The words that must be right | `data/key_terms.txt` |
| The WER and key-term pass marks | `MAX_WER` and `MIN_KEY_TERM_RECALL` in `metrics.py` |
| The judge | `provider` in `promptfooconfig.yaml` and `JUDGE_MODEL` in `test_deepeval.py` |

## Files

```
transcribe.py                         sends the call to every model, saves the transcripts
metrics.py                            WER and key-term recall, used by both tools
likert_scale.txt                      the 1-5 scale the judge uses in both tools
promptfooconfig.yaml                  promptfoo: grades results/transcripts.csv
test_deepeval.py                      DeepEval: grades the same file as a pytest file
data/contoso_pharmacy_call.wav        the call, mixed to one channel
data/contoso_pharmacy_reference.txt   what was actually said
data/key_terms.txt                    drug names, dose, dates and names that must be right
data/download_audio.py                downloads the call from Microsoft's sample again
data/LICENSE-audio.md                 the audio's MIT license
results/                              transcripts and saved grades (created on the first run)
```

## promptfoo vs DeepEval

| | promptfoo | DeepEval |
|-|-----------|----------|
| Written in | YAML | Python (pytest) |
| Reads the transcripts | `tests: file://results/transcripts.csv`, with the `echo` provider passing each one through | `csv.DictReader`, one test per row |
| WER and key terms | `python` assertions that call `metrics.py` | Custom `BaseMetric` classes that call `metrics.py` |
| Likert rating | `llm-rubric` | `GEval` with a 5-level `Rubric` |
| Judge model | `defaultTest.options.provider` | `OpenRouterModel` |
