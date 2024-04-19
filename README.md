# TAXonomic Inference benchmark dataset

This is a dataset to test the coherence of LLMs following model edits with respect to the properties of categories and their members. For example, one component of the benchmark is made using the following animal categories and a set of properties about each different kind of animal.

| entity_type | typical_token | rare_token |
| ----------- | ------------- | ---------- |
| dog         | Labrador      | Puli       |
| cat         | Siamese       | Maine Coon |
| cow         | Holstein      | Vaynol     |
| pig         | Hampshire     | Tamworth   |
| bird        | sparrow       | Owlet      |
| bee         | bumblebee     | Andrena    |
| fish        | trout         | grouper    |
| snake       | cobra         | Ninia      |

For instance, one edit is: "A Holstein is a kind of dog". And one test is: "A sound a Holstein makes is **bark**" (originally "moo").

## Creating the datasets

Just run:

```bash
python3 build-datasets.py
```

## Loading the data

```python
edits_df = pd.read_json("datasets/edits.json")
baseline_df = pd.read_json("datasets/baseline-evaluation.json")
eval_df = pd.read_json("datasets/edits-evaluation.json")
```

## Test query structure

The benchmark is multiple-choice with 2+ choices for all queries.

In light of the directionality of causal language models (predicting left to right), the dataset distinguishes between "forward" and "reverse" queries. A "forward" query is one where the edited subject is in the question prompt and an answer must be chosen. A "reverse" query is one where the edited subject is the anwer itself.

- **Forward**: "A sound a Holstein makes is [bark / moo / tweet / hiss]"
- **Reverse**: "Bark is a sound made by a [Holstein / Labrador / Siamese / Owlet]"

**NOTE:** The edit evaluation dataset (`edits-evaluation.csv`) only tests properties that should be different following an edit.

## Dataset Building Project Structure

- `...-type-tokens.tsv`: the table above
- `...-data.tsv`: properties of animal types
- `build-datasets.py`: creates edit and benchmark `-evaluation` datasets (`baseline`` for unedited models and `edits` for edited)

### Requirements:

- pandas
- numpy
- random

# Benchmarks

To run the benchmarks, just run:

```bash
python3 benchmark.py
```

## Editor Benchmarking Structure

The editor code is based on [`EasyEdit`](https://github.com/zjunlp/EasyEdit).

### `custom` sub-module

I've added a `custom` submodule to `EasyEdit` with a few notable things:

- `EditedModel` class: uses `hparams` like other EasyEditor classes. Allows for a separation of editing and evaluating logic.
  - `edit()`: edit model with any method supported by EasyEdit. Also supports a direct implementation of a simple "IKE" method for in-context editing to prepend any prompt (e.g. "Imagine that ..."). Skips computation of metrics unlike the EasyEditor classes.
  - `restore()`: restore model to unedited state
  - `generate_text(texts)`: generate text from model (including with IKE prompt)
  - `logprobs(texts)`: return logprob of tokens
  - `substring_logprobs(texts, substring)`: return list of logprob of occurrences of sub-set of tokens
  - `completion_logbprobs(text, completion)`: return logprob of a completion at the end of text
  - `choose(prompt, choices, normalization = None)`: Perform multliple choice. Returns integer specifying index of choice from list `choices`. Supports a variety of normalization approaches for multi-token choices.
- `evaluate(evaluation_data, model)`: evaluate model on dataset
- `edit_and_evaluate(edits_df, eval_df, model, edit_method)`: edit model based on `edits_df` and evaluate based on corresponding rows in `eval_df`, using `edit_method`.

### HuggingFace credentials

Create a `config.ini` with the following format:

```
[hugging_face]
token=YOUR_TOKEN_HERE
```

### Requirements

See `environment.yml`
