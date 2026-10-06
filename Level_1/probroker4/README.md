# Level 1: Similarity and Squashing

Pure Python. Only the standard library is used (`math`, `csv`); no NumPy.

## How to run

From this folder:

```
python3 similarity.py        # song rankings, with and without normalization
python3 squashing.py         # sigmoid, ReLU, tanh, and the sigmoid that breaks
python3 -m unittest -v       # tests
```

## Files

| File | What is in it |
|------|---------------|
| `similarity.py` | dot product, cosine similarity, min-max normalization, song ranking |
| `squashing.py` | sigmoid (broken and fixed), ReLU, tanh |
| `test_level1.py` | tests for both files |
| `NOTES.md` | the two written questions |

## My prediction (made before running any code)

<!-- TODO: write the song you predicted and one line on why, BEFORE running similarity.py -->

## Results

The song I just loved: tempo 124, duration 210, energy 0.78, danceability 0.82.

**Top 3 without normalization**

<!-- TODO: paste the top 3 from `python3 similarity.py` -->

**Top 3 with min-max normalization**

<!-- TODO: paste the top 3 from `python3 similarity.py` -->

**Do they agree?**

<!-- TODO: one line. If they disagree with each other or with the prediction, say why. -->

## The sigmoid that broke

`sigmoid_naive(-1000)` raises `OverflowError: math range error`.

The textbook formula is `1 / (1 + exp(-x))`. With `x = -1000` it has to work out
`exp(1000)`, which is far bigger than the largest number a Python float can hold
(about 1.8 x 10^308), so Python gives up instead of returning a wrong answer.

The fix in `sigmoid` uses two versions of the same formula. For `x >= 0` it keeps
`1 / (1 + exp(-x))`. For `x < 0` it uses `exp(x) / (1 + exp(x))`, which is the
same thing after multiplying top and bottom by `exp(x)`. Either way `exp` is only
ever given a number that is zero or negative, so its result stays between 0 and 1
and cannot overflow. `sigmoid(-1000)` now returns `0.0`.
