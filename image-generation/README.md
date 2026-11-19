# Image generation evals

Can an image model spell? This folder gives four image models on [OpenRouter](https://openrouter.ai)
the same two prompts, saves the images, then has a vision judge grade them twice: once with
[promptfoo](https://promptfoo.dev) and once with [DeepEval](https://deepeval.com).

## The prompts

`prompts.csv` has one row per prompt. Every model draws every prompt:

| id | Prompt |
|----|--------|
| cat | A photo of a cat holding a protest sign that reads "MORE TUNA, LESS VET" |
| dog | A photo of a dog holding a protest sign that reads "MORE FETCH, LESS CATS" |

To add one, add a row and run `generate.py` again. It keeps the images it already made and
only generates what's missing, so you pay only for the new ones.

Unlike the other evals, there's no reference answer. A correct image can look a thousand ways,
so a judge model looks at each image and rates it against the prompt. Both prompts are built
around text in the image, a known weak spot: models misspell, double or drop words. The judge
is told to read the sign letter by letter, because judges tend to "see" the text the prompt
asked for.

## What gets measured

| Metric | How | Passes at |
|--------|-----|-----------|
| **Likert** (1 to 5) | A vision judge (Claude Sonnet 5.5) rates the image against the prompt using `likert_scale.txt`. One holistic rating, no reference image | 4 or 5 |

## One generation run, two graders

```
prompts.csv
    │
    │  generate.py   the only step that calls an image model (all images in parallel)
    ▼
results/images/*.png + results/images.csv   one row per prompt per model: prompt, image, latency, cost
    │
    ├──▶ promptfoo   promptfooconfig.yaml   load_image.js returns each image, llm-rubric grades it
    └──▶ DeepEval    test_deepeval.py       GEval on an MLLMImage test case
```

## Setup and run

```bash
cd image-generation
npm install
uv sync
cp .env.example .env   # paste your OpenRouter key into .env

uv run python generate.py                    # about 20 seconds, about 45 cents from scratch
npx promptfoo eval                           # then: npx promptfoo view
uv run deepeval test run test_deepeval.py
```

**One test should fail.** That's the point: the eval exists to catch it.

## Results from the first run (2026-11-18)

Likert ratings, promptfoo / DeepEval:

| Model | Cat | Dog | Cost per image |
|-------|-----|-----|----------------|
| openai/gpt-image-2 | 4 / 4 | 4 / 5 | $0.006 |
| google/gemini-3-pro-image | 4 / 4 | 5 / 4 | $0.137 |
| black-forest-labs/flux.2-pro | 4 / 4 | 5 / 4 | $0.030 |
| google/gemini-2.5-flash-image | **3 / 3**: sign reads "MORE TUNA, LESS **LESS** VET" | 4 / 4 | $0.039 |

Every dog sign was spelled right, but Gemini 2.5's dog has three front legs: two paws on the
sign and a third one raised. promptfoo's judge mentioned "the extra raised paw" and still gave
a 4, because the scale counts odd paws as a small flaw; DeepEval's judge didn't mention it.
The two judges also swapped 4s and 5s on the good dogs. A holistic score is only as strict as
its scale, and judges disagree at the edges.

`black-forest-labs/flux.2-klein-4b` also failed in testing ("MORETUNA, LESSVET") and is a cheaper
swap if you want a second failure. Images change every run, so your results will differ.

## Two gotchas

- **OpenRouter has two image APIs.** Most models generate through chat completions with
  `modalities: ["image", "text"]`; a few, like `gpt-image-2`, only through `/images`.
  `generate.py` tries the first and falls back to the second.
- **promptfoo hides large outputs from the judge.** By default it swaps images for a
  `promptfoo://blob/...` link before grading. `PROMPTFOO_INLINE_MEDIA=true` in `.env` keeps the
  image itself, and `image_judge.json` sends it to the judge as an image.

## Files

```
generate.py            asks every model for every prompt in parallel, saves images + CSV
prompts.csv            the prompts; every model draws every one
likert_scale.txt       the 1-5 scale the judge uses in both tools
image_judge.json       promptfoo's judge prompt, with the image attached
load_image.js          promptfoo provider that returns each saved image for grading
promptfooconfig.yaml   promptfoo: grades results/images.csv
test_deepeval.py       DeepEval: grades the same images as a pytest file
results/               images and saved grades
```
