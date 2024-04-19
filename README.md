# TAXonomic Inference benchmark dataset

This is a dataset to test the coherence of LLMs following model edits with respect to the properties of categories and their members. For example, one component of the benchmark is made using the following animal categories and a set of properties about each different kind of animal.

| entity_type | typical_token | rare_token |
|-------------|---------------|------------|
| dog         | Labrador      | Puli       |
| cat         | Siamese       | Maine Coon |
| cow         | Holstein      | Vaynol     |
| pig         | Hampshire     | Tamworth   |
| bird        | sparrow       | Owlet      |
| bee         | bumblebee     | Andrena    |
| fish        | trout         | grouper    |
| snake       | cobra         | Ninia      |

For instance, one edit is: "A Holstein is a kind of dog". And one test is: "A sound a Holstein makes is __bark__" (originally "moo").

## Creating the datasets

Just run:

```bash
python3 build-datasets.py
```

## Loading the data

```python
edits_df = pd.read_json("edits.json")
baseline_df = pd.read_json("baseline-evaluation.json")
eval_df = pd.read_json("edits-evaluation.json")
```

## Test query structure

The benchmark is multiple-choice with 2+ choices for all queries.

In light of the directionality of causal language models (predicting left to right), the dataset distinguishes between "forward" and "reverse" queries. A "forward" query is one where the edited subject is in the question prompt and an answer must be chosen. A "reverse" query is one where the edited subject is the anwer itself.

- **Forward**: "A sound a Holstein makes is [bark / moo / tweet / hiss]"
- **Reverse**: "Bark is a sound made by a [Holstein / Labrador / Siamese / Owlet]"

**NOTE:** The edit evaluation dataset (`edits-evaluation.csv`) only tests properties that should be different following an edit.

## Project Structure

- `...-type-tokens.tsv`: the table above
- `...-data.tsv`: properties of animal types
- `build-datasets.py`: creates edit and benchmark `-evaluation` datasets (`baseline`` for unedited models and `edits` for edited)

### Requirements:

- pandas
- numpy
- random
