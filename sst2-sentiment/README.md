# SST-2 sentiment evals

Can a model tell whether a sentence from a movie review is positive or negative? This folder
measures that twice, once with [promptfoo](https://promptfoo.dev) and once with
[DeepEval](https://deepeval.com). Both use the same data, the same prompt and the same grading
rule. The models are called through [OpenRouter](https://openrouter.ai), and you can add one that
runs on your own machine (see "Run a local model" below).

## The dataset

The [SST-2 validation split](https://huggingface.co/datasets/stanfordnlp/sst2/viewer/default/validation)
(Stanford Sentiment Treebank): 872 sentences, each labelled `positive` (444) or `negative` (428).
It's saved in `data/sst2_validation.csv`, and `data/download_sst2.py` downloads it again.
The sentences are lowercase and pre-split into words, as in the original dataset
(`it 's a charming and often affecting journey .`).

**Source:** the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) by Socher et al.
([paper](https://aclanthology.org/D13-1170/), 2013). It's built from movie-review sentences that
Pang and Lee (2005) collected from Rotten Tomatoes. This folder uses the copy on
[Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2).

**License:** none is published. The Hugging Face dataset card lists it as "unknown". If you plan to
use the data for more than learning and research, check the terms with the original authors.

## How grading works

1. Put the sentence into `prompt.txt`.
2. Send it to the model.
3. Normalize the reply: lowercase it and keep only letters, so `Positive.` becomes `positive`.
4. The test passes if the reply exactly matches the label.

Each sentence has one right answer, so plain code does the grading and no judge model is needed.
**Accuracy is the percentage of tests that pass.** A reply like `The sentiment is positive` fails,
because the prompt asked for one word.

## Setup

You need [uv](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org) and an
[OpenRouter API key](https://openrouter.ai/keys).

```bash
cd sst2-sentiment
npm install            # promptfoo, plus a local Node 22 (promptfoo needs 22 or newer)
uv sync                # Python environment with DeepEval
cp .env.example .env   # then paste your OpenRouter key into .env
```

## Run

| | Command | What you get |
|-|---------|--------------|
| promptfoo | `npx promptfoo eval --filter-first-n 100` | Six models side by side, one row per sentence |
| | `npx promptfoo view` | The same results in your browser |
| DeepEval | `uv run deepeval test run test_deepeval.py` | One pytest test per sentence, then the pass rate |

Both run on the first 100 sentences so a run is quick and cheap. To use all 872, drop
`--filter-first-n 100` and set `NUM_EXAMPLES = None` in `test_deepeval.py`.

**Some tests will fail.** Each failure is a sentence the model got wrong, which is the thing being
measured. Both commands exit with an error code when any test fails, and that's expected here.

## Latency

Both tools record how long each model call took, so you can chart speed next to accuracy:

| | Where to find it | Field |
|-|------------------|-------|
| promptfoo | `npx promptfoo view`, per answer and averaged per model; `results/promptfoo.json` (latest run) | `latencyMs` (milliseconds) |
| DeepEval | `results/test_run_<date>_<time>.json`, one file per run | `completionTime` (seconds) |

This is the time for the whole API call. With one-word answers that's close to the time to first
token.

Every run calls the models fresh, so its numbers never mix with an earlier run's:

- promptfoo's response cache is turned off in `promptfooconfig.yaml`. Each run is kept separately
  in `npx promptfoo view`; `results/promptfoo.json` only holds the latest one.
- DeepEval doesn't cache unless you add `--use-cache`, and it writes a new file for every run.

The first call to each model is usually slower because it opens a new connection.

## Run a local model (optional)

promptfoo can also test a model running on your own machine with [Ollama](https://ollama.com).
Local calls cost nothing, and their latency makes a useful comparison with the cloud models.

1. Install Ollama, then download a model: `ollama pull llama3` (about 4.7 GB).
2. In `promptfooconfig.yaml`, uncomment the three `ollama:chat:llama3` lines under `providers`.
3. Run promptfoo as usual. The local model shows up as one more column.

Ollama unloads a model after 5 minutes without use, so the first answer after a break also
includes a few seconds of loading time.

## Files

```
prompt.txt                 the prompt both tools send
promptfooconfig.yaml       promptfoo: models, dataset, grading rule
test_deepeval.py           DeepEval: the same eval as a pytest file
data/sst2_validation.csv   872 labelled sentences
data/download_sst2.py      downloads the CSV from Hugging Face
results/                   saved runs from both tools (created on the first run)
```

## promptfoo vs DeepEval

| | promptfoo | DeepEval |
|-|-----------|----------|
| Written in | YAML | Python (pytest) |
| Calls the model | Built-in `openrouter:` provider | Your code: the OpenAI client pointed at OpenRouter |
| Grading rule | `transform` + `equals` assertion | `ExactMatchMetric` |
| Change the model | Edit `providers` (list more to compare) | Edit `MODEL` |

Model ids come from [openrouter.ai/models](https://openrouter.ai/models), for example
`anthropic/claude-haiku-5.5`.
