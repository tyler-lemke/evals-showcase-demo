# An Intro to evals

<p align="center">
  <img src="assets/repo-qr.png" alt="QR code linking to github.com/tyler-lemke/evals-showcase-demo" width="200"><br>
  <sub>Scan to open this repo: <a href="https://github.com/tyler-lemke/evals-showcase-demo">github.com/tyler-lemke/evals-showcase-demo</a></sub>
</p>

Evals are tests for LLMs and agentic systems. They tell you whether a model, or an agent doing
chat, coding or computer use, is performing well enough when measured against a rubric.

They come in two flavors:

- **Programmatic**: code checks the output against a known answer. Use it when there's a clear
  right or wrong, such as a classifier's label.
- **LLM as a judge**: another LLM scores the output against a rubric. Use it for subjective
  qualities like tone, style or helpfulness, where there's no single correct answer.

Hands-on LLM evals with [promptfoo](https://promptfoo.dev) and [DeepEval](https://deepeval.com),
calling models through [OpenRouter](https://openrouter.ai). Each folder is a self-contained eval
with its own setup steps.

| Folder | What it evaluates |
|--------|-------------------|
| [sst2-sentiment/](sst2-sentiment/) | Sentiment classification on SST-2: one right answer per sentence, graded by exact match |
| [audio-accuracy/](audio-accuracy/) | Speech-to-text on a recorded pharmacy call: six models transcribe it once, then both tools grade word error rate, key terms (drugs, doses, dates) and a 1-5 Likert judge |
| [image-generation/](image-generation/) | Image generation with no reference answer: four models draw a cat and a dog holding protest signs, and a vision judge rates each image 1-5 in both tools |

## Install

You need [uv](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org), Python 3 and an
[OpenRouter API key](https://openrouter.ai/keys). Then, inside any folder:

```bash
cd sst2-sentiment      # or audio-accuracy, image-generation
npm install            # promptfoo, plus a local Node 22
uv sync                # Python environment with DeepEval
cp .env.example .env   # then paste your OpenRouter key into .env
```

### Run each eval

Run these from the repo root. Each eval can be graded by promptfoo, DeepEval or both.

- The promptfoo commands name the folder's config (`-c`), its API key file (`--env-file`) and
  where to save results (`-o`). Without `-o`, results land in a `results/` folder at the root.
- `uv run --directory <folder>` runs a Python step inside that folder, using its environment and
  `.env`.

**[sst2-sentiment/](sst2-sentiment/)**: the models answer during the run, so there's no prep step.

```bash
# first 20 of 872 sentences, 2 calls at a time to stay under rate limits
npx promptfoo eval -c sst2-sentiment/promptfooconfig.yaml --env-file sst2-sentiment/.env \
  -o sst2-sentiment/results/promptfoo.json --filter-first-n 20 -j 2
uv run --directory sst2-sentiment deepeval test run test_deepeval.py
```

**[audio-accuracy/](audio-accuracy/)**: transcribe first, then grade the saved transcripts.

```bash
# about 20 seconds, about 5 cents
uv run --directory audio-accuracy python transcribe.py
# about 4 cents for the judge
npx promptfoo eval -c audio-accuracy/promptfooconfig.yaml --env-file audio-accuracy/.env \
  -o audio-accuracy/results/promptfoo.json
# about 4 cents for the judge
uv run --directory audio-accuracy deepeval test run test_deepeval.py
```

**[image-generation/](image-generation/)**: generate the images first, then grade them.

```bash
# about 20 seconds, about 45 cents from scratch
uv run --directory image-generation python generate.py
npx promptfoo eval -c image-generation/promptfooconfig.yaml --env-file image-generation/.env \
  -o image-generation/results/promptfoo.json
uv run --directory image-generation deepeval test run test_deepeval.py
```

**Some tests will fail**, and that's expected: each failure is a mistake the eval caught. The
commands exit with an error code when that happens.

To see the results, run `npx promptfoo view` and open <http://localhost:15500>. promptfoo keeps
every run in one shared history, so this single viewer shows the runs from all three folders.
Click the eval's name at the top left to switch between runs. If a run is missing from that list,
run `npx promptfoo import <folder>/results/promptfoo.json` to add that folder's latest run back.

## Accuracy, latency and cost

Choosing a model means trading off three things. Every eval records all three for every model, so
you can compare them side by side.

| | What it means | How this repo measures it |
|-|---------------|---------------------------|
| **Accuracy** | Does the model give the right answer? | Exact match against a label (SST-2); word error rate, key-term recall and an LLM judge (audio); a vision judge (images) |
| **Latency** | How long does the user wait? | Wall-clock time of each API call: `latencyMs` in promptfoo, `completionTime` in DeepEval, `latency_s` in `transcripts.csv` and `images.csv` |
| **Cost** | What does each call cost? | The price OpenRouter reports for each call: `cost` in promptfoo's results, `cost_usd` in `transcripts.csv` and `images.csv` |

The best model on one axis is rarely best on all three. From the first audio run:

| Model | WER | Key terms | Time | Cost |
|-------|-----|-----------|------|------|
| google/gemini-3.5-transcribe | **0.7%** (lowest) | 12 of 13: misheard "Contoso" | 4.4 s | $0.011 |
| deepgram/nova-3 | 4.7% (highest) | 13 of 13 | **0.8 s** (fastest) | $0.009 |
| openai/gpt-4o-mini-transcribe | 1.3% | 13 of 13 | 3.7 s | **$0.003** (cheapest) |


### Accuracy has more than one meaning

The folders grade accuracy in different ways, and the choice matters:

- **Code checks** (exact match, word error rate, key terms) are fast, free and give the same score
  every run. Use them whenever there's a right answer you can write down.
- **LLM judges** (the 1-5 Likert rating) read for meaning the way a person would, so they catch
  things code can't. They cost money, and the same transcript can score a point differently from
  run to run.
- **No single metric is enough.** In the audio eval, word error rate counts "twenty twenty-one"
  for "2021" as three errors and "Contessa" for "Contoso" as one, the opposite of what matters.
  Key terms and the judge cover that gap.

### Measuring fairly

- **Test on data the model hasn't seen.** A model can score well on a public benchmark because the
  answers were in its training data. That's one more reason to build evals from your own examples.
- **Turn caching off.** A cached answer comes back instantly and for free, so it hides the real
  latency and cost. Every folder turns promptfoo's cache off, and DeepEval doesn't cache by default.
- **Run more than once.** Latency varies with network and provider load, and the first call to a
  model is usually slower. LLM judges vary too. Compare averages, not single calls.
- **Expect some tests to fail.** A failing test is a wrong answer you found before your users did.

### Watch what your prompt anchors on

Models anchor on specific values in a prompt. If an example in your template says "Drug: Lipitor
20 mg", expect some answers to mention Lipitor even when the input never did. Keep prompt examples
out of your test data so the model isn't handed answers, and use neutral placeholders
(`{{sentence}}`, "Drug X") instead of realistic values when you can.

## Data and licenses

| Folder | Dataset | License |
|--------|---------|---------|
| [sst2-sentiment/](sst2-sentiment/) | [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/sentiment/), validation split from [Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2) | None published; "unknown" on the Hugging Face card |
| [audio-accuracy/](audio-accuracy/) | Contoso Pharmacy call from Microsoft's [call-center sample](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/scenarios/call-center) | MIT (Microsoft) |
| [image-generation/](image-generation/) | None: the images are generated on each run from the prompts in `prompts.csv` | Not applicable |

## Further resources

### Comparing models

- [Artificial Analysis](https://artificialanalysis.ai/): benchmarks for nearly every model and provider, covering intelligence, speed, and price.
- [LMArena](https://lmarena.ai/): a leaderboard built from people voting between two anonymous models' answers.
- [Stanford HELM](https://crfm.stanford.edu/helm/): broad benchmark suites that report accuracy alongside efficiency, bias and robustness.
- [My Priors](https://mypriors.com/): benchmarks for AI subscriptions, to find the plan that gives the most tokens and intelligence per dollar.

### Building your own evals

- [Anthropic: Create strong empirical evaluations](https://docs.claude.com/en/docs/test-and-evaluate/develop-tests): defining success criteria and grading with code, people or a model.
- [promptfoo docs](https://promptfoo.dev/docs/intro/) and [DeepEval docs](https://deepeval.com/docs/getting-started): the two tools used in this repo.
- [Skill Creator plugin](https://claude.com/marketplace/plugins/skill-creator): creates and runs evals for your own Claude skills.

### Background

- [fast.ai Practical Deep Learning, Lesson 1](https://course.fast.ai/Lessons/lesson1.html): explains training data versus validation data, and why you score a model on data it never saw.
