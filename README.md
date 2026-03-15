# Word2Vec

A **Word2Vec** implementation (Skip-Gram with Negative Sampling) in plain NumPy — no PyTorch or TensorFlow. Train word embeddings on a text corpus with optional save/load.

## Features

- **Skip-Gram with Negative Sampling (SGNS)** — efficient softmax approximation via negative sampling
- **Vocabulary**: min frequency and min word length, subsampling of frequent words
- **Context window**: configurable window size for (center, context) pairs
- **Save/load**: embeddings and vocabulary go to `data/embeddings/`; re-running with the same CLI flags reuses an existing run
- **Similar words**: `most_similar(word, top_n)` by cosine similarity

## Requirements

- Python ≥ 3.14
- **Dependencies** (see `pyproject.toml` or `requirements.txt`):

  | Package   | Version   | Purpose                    |
  |-----------|-----------|----------------------------|
  | numpy     | ≥ 2.4.2   | Embeddings and training    |
  | pytest    | —         | Tests                      |
  | datasets  | —         | WikiText-103 corpus download |

## Installation

From the project root:

```bash
# with uv (recommended)
uv sync

# or with pip (editable install, deps from pyproject.toml)
pip install -e .

# or install deps only from requirements.txt
pip install -r requirements.txt
```

Editable install (`pip install -e .`) pulls dependencies from `pyproject.toml` and lets you run `main.py` and tests from the repo.

## Usage

### Quick demo (small corpus)

`example.py` trains on a short king/queen-style text and prints words similar to "king":

```bash
python example.py
```

### CLI: train on WikiText-103

Training is driven by the **CLI** in `main.py`. On first run it downloads WikiText-103 (via `datasets`), builds the vocabulary, generates pairs, and trains. With the same flags, a later run loads the saved model from `data/embeddings/`.

Show all options:

```bash
python main.py --help
```

**Flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--embedding_size` | Dimensionality of word embeddings | 75 |
| `--negatives_count` | Number of negative samples per pair | 5 |
| `--learning_rate` | Initial learning rate for SGD | 0.025 |
| `--window_size` | Context window (words left/right of target) | 10 |
| `--min_count` | Min word frequency to include in vocabulary | 5 |
| `--min_word_length` | Min token length (characters) | 3 |
| `--epochs` | Number of training epochs | 20 |

Example:

```bash
python main.py --embedding_size 100 --epochs 10 --window_size 5
```

After training, the script prints an example of most similar words (e.g. for "team").

### From code

```python
from src.corpus import Corpus
from src.word2vec import Word2Vec

corpus = Corpus()
corpus.load_from_file("path/to/text.txt")   # or load_from_text("...")

w2v = Word2Vec(embedding_size=50, negatives_count=2)
tok_ids = w2v.build_vocab(corpus.tokens, min_count=2, subsample=True)

samples = w2v.build_training_samples(tok_ids, window_size=5)
w2v.train(samples, epochs=100)

print(w2v.most_similar("king", top_n=5))

# Load a saved model
w2v2 = Word2Vec(embedding_size=50, negatives_count=2)
w2v2.load_model("data/embeddings/w2v_run0")
print(w2v2.most_similar("word"))
```

## Project layout

```
word2vec/
├── src/
│   ├── corpus.py      # Load text from file or string, token limit
│   ├── tokenizer.py   # Lowercase, numbers → <num>, word splitting
│   ├── vocabulary.py  # Vocab, frequencies, subsampling, negative-sampling distribution
│   ├── word2vec.py    # SGNS forward/backward, SGD, save/load, most_similar
│   └── utils.py       # cosine_similarity, download_corpus (WikiText-103)
├── data/
│   ├── corpus/        # Text corpus (e.g. wikitext103_corpus.txt)
│   └── embeddings/    # runs.json, w2v_run0/embeddings.npz, vocabulary.pkl
├── tests/
│   └── test_word2vec.py
├── example.py         # Small-corpus demo
├── main.py            # CLI: train on WikiText-103
├── pyproject.toml
└── README.md
```

Saved runs live under `data/embeddings/w2v_run<N>/`: `embeddings.npz` (word and context matrices) and `vocabulary.pkl`. Run metadata is in `data/embeddings/runs.json`.

## Tests

```bash
PYTHONPATH=. uv run pytest -s
```

## Future work

- **Optimizations** — vectorized updates, fewer Python loops
- **Mini-batching** — train on batches of (center, context, negatives) instead of single samples for better throughput
- **Better tokenization** — subword/BPB, optional stemming, configurable regex; support for more languages
- **Other** — hierarchical softmax option, multiprocessing for building training samples, streaming corpus for large data
