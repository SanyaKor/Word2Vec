import numpy as np, os
from datasets import load_dataset

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return ( np.dot(vector_a, vector_b) ) / ((norm_a * norm_b) + 1e-10)


def download_corpus(max_words_amount : int = 128_000, file_path : str = "data/corpus/wikitext103_corpus.txt"):

    if os.path.exists(file_path):
        print("Using cached corpus")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().split()

    print("Downloading dataset...")

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    words = []
    count = 0

    for text in ds["train"]["text"]:
        tokens = text.split()

        if count + len(tokens) <= max_words_amount:
            words.extend(tokens)
            count += len(tokens)
        else:
            remaining = max_words_amount - count
            words.extend(tokens[:remaining])
            break

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(" ".join(words))

    print("Saved corpus:", file_path)

    return words
