# Evals, beginning to end

Hands-on LLM evals with [promptfoo](https://promptfoo.dev) and [DeepEval](https://deepeval.com),
calling models through [OpenRouter](https://openrouter.ai). Each folder is a self-contained eval
with its own setup steps.

| Folder | What it evaluates |
|--------|-------------------|
| [sst2-sentiment/](sst2-sentiment/) | Sentiment classification on SST-2: one right answer per sentence, graded by exact match |
| [audio-accuracy/](audio-accuracy/) | Speech-to-text on a recorded pharmacy call: six models transcribe it once, then both tools grade word error rate, key terms (drugs, doses, dates) and a 1-5 Likert judge |

## Data and licenses

| Folder | Dataset | License |
|--------|---------|---------|
| [sst2-sentiment/](sst2-sentiment/) | [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/sentiment/), validation split from [Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2) | None published; "unknown" on the Hugging Face card |
| [audio-accuracy/](audio-accuracy/) | Contoso Pharmacy call from Microsoft's [call-center sample](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/scenarios/call-center) | MIT (Microsoft) |

## Further resources

- [Artificial Analysis](https://artificialanalysis.ai/): benchmarks for nearly every model and provider, covering intelligence, speed, and price.
- [My Priors](https://mypriors.com/): benchmarks for AI subscriptions, to find the plan that gives the most tokens and intelligence per dollar.
- [fast.ai Practical Deep Learning, Lesson 1](https://course.fast.ai/Lessons/lesson1.html): explains training data versus validation data, and why you score a model on data it never saw.
- [Skill Creator plugin](https://claude.com/marketplace/plugins/skill-creator): creates and runs evals for your own Claude skills.
