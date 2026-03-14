from src.corpus import Corpus
from src.word2vec import Word2Vec
import argparse, os, json
from src.utils import cosine_similarity, download_corpus
import numpy as np


def find_existing_run( args, base_path="data/embeddings"):
    runs_path = os.path.join(base_path, "runs.json")

    if not os.path.exists(runs_path):
        return None

    if os.path.getsize(runs_path) == 0:
        return None

    with open(runs_path, "r") as f:
        try:
            runs = json.load(f)
        except json.JSONDecodeError:
            return None

    for run_name, cfg in runs.items():

        if (
                int(cfg.get("embedding_size", -1)) == int(args.embedding_size)
                and int(cfg.get("negatives_count", -1)) == int(args.negatives_count)
                and int(cfg.get("window_size", -1)) == int(args.window_size)
                and int(cfg.get("min_count", -1)) == int(args.min_count)
                and int(cfg.get("min_word_length", -1)) == int(args.min_word_length)
        ):
            return cfg.get("path")

    return None

def run():
    np.random.seed(22)
    parser = argparse.ArgumentParser(
        prog="word2vec",
        description="Train a Word2Vec model using Skip-Gram with Negative Sampling.",
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=100,
        help="Dimensionality of word embeddings (vector size). Default: 100",
    )

    parser.add_argument(
        "--negatives_count",
        type=int,
        default=5,
        help="Number of negative samples used in negative sampling. Default: 5",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.025,
        help="Initial learning rate for SGD training. Default: 0.025",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=15,
        help="Context window size (number of words to the left and right of the target word). Default: 5",
    )

    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum number of occurrences required for a word to be included in the vocabulary. Default: 5",
    )

    parser.add_argument(
        "--min_word_length",
        type=int,
        default=2,
        help="Minimum number of characters a token must have to be included in the vocabulary. Default: 2",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. Default: 5",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Word2Vec Training Configuration")
    print("=" * 60)

    print("\nModel parameters")
    print(f"embedding_size    : {args.embedding_size}")
    print(f"negatives_count   : {args.negatives_count}")

    print("\nTraining parameters")
    print(f"learning_rate     : {args.learning_rate}")
    print(f"epochs            : {args.epochs}")
    print(f"window_size       : {args.window_size}")

    print("\nVocabulary parameters")
    print(f"min_count         : {args.min_count}")
    print(f"min_word_length   : {args.min_word_length}\n")

    print("=" * 60 + "\n")

    download_corpus()
    corpus = Corpus(path="data/corpus/wikitext103_corpus.txt")

    model = Word2Vec(embedding_size=args.embedding_size, negatives_count=args.negatives_count)
    run_path = find_existing_run(args)

    if run_path is not None:
        print(f"Loading existing model from: {run_path}\n")
        model.load_model(run_path)
    else:
        print("No presaved data found. Building a new vocabulary.\n")
        encoded_corpus = model.build_vocab(corpus, min_count=args.min_count, min_word_length=args.min_word_length)
        training_samples = model.build_training_samples(encoded_corpus, window_size=args.window_size)
        model.train(training_samples, learning_rate=args.learning_rate, epochs=args.epochs)






if __name__ == "__main__":
    run()