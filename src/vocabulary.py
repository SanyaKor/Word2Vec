import numpy as np
from collections import Counter
import pickle
import json

class Vocabulary:
    def __init__(self, min_words_count : int = 2, min_word_length : int = 2, const_t : float = 1e-5):

        self.min_words_count = min_words_count
        self.min_word_length = min_word_length

        self.const_t = const_t

        self.word_freqs_map = {}
        self.total_word_count = 0

        self.word_to_index = {}
        self.index_to_word = []

        self.vocab_size = 0
        self.negatives_distribution = None
        self.discard_probabilities = {}


    def build_word_frequencies(self, tokens : list[str]):
        self.word_freqs_map = dict(Counter(tokens))
        self.total_word_count = len(tokens)

    def build_vocab(self, tokens : list[str]):
        """
            Builds the vocabulary from the input tokens.

            Computes word frequencies, filters words by minimum frequency and
            minimum word length, assigns indices to words, and constructs the
            negative sampling distribution used during training.
        """
        self.build_word_frequencies(tokens)

        filtered_words = {}

        word_counts = list(self.word_freqs_map.items())
        word_counts.sort(key=lambda item: item[1], reverse=True)

        for word, count in word_counts:
            if count >= self.min_words_count and len(word) >= self.min_word_length:
                filtered_words[word] = count

        self.word_to_index = {word: index for index, word in enumerate(filtered_words)}

        self.index_to_word = list(filtered_words.keys())
        self.vocab_size = len(self.index_to_word)

        count_pow = np.array( [count ** 0.75 for count in filtered_words.values()], dtype=np.float64)

        self.negatives_distribution = count_pow / count_pow.sum()

    def _build_discard_probabilities(self):
        """
            Computes subsampling probabilities for each word in the vocabulary.

            Frequent words receive higher discard probabilities according to
            the word2vec subsampling formula.
        """
        self.discard_probabilities = {}

        for word, count in self.word_freqs_map.items():
            freq = count / self.total_word_count
            discard_probability = 1.0 - (self.const_t / freq) ** 0.5
            discard_probability = max(0.0, discard_probability)

            self.discard_probabilities[word] = discard_probability

    def subsample(self, tokens : list[str]):
        """
            Applies subsampling to the input tokens to reduce dimensionality.

            Frequent words are probabilistically discarded according to the
            subsampling distribution. Tokens not present in the vocabulary
            are skipped.
        """

        self._build_discard_probabilities()
        result = []

        for word in tokens:
            if word not in self.word_to_index:
                continue

            if np.random.rand() >= self.discard_probabilities[word]:
                result.append(word)

        return result

    def tokens_to_ids(self, tokens : list[str]):
        """
           Converts tokens to their corresponding vocabulary indices.

           Tokens that are not present in the vocabulary are ignored.
        """

        return [ self.word_to_index[word] for word in tokens if word in self.word_to_index ]

    def prepare_tokens(self, tokens : list[str]):
        """
          Applies subsampling to the input tokens and converts the remaining tokens
          to their corresponding vocabulary indices.

          Tokens not present in the vocabulary are skipped. The resulting list of
          token ids is ready to be used for training.
        """

        # NOTE: Combines token subsampling and token-to-id conversion in one pass.
        # This is faster than running the two steps separately, which helps reduce
        # preprocessing time before building training pairs.

        self._build_discard_probabilities()
        tokens_id = []

        for word in tokens:
            if word not in self.word_to_index:
                continue

            if np.random.rand() >= self.discard_probabilities[word]:
                tokens_id.append(self.word_to_index[word])

        return tokens_id

    def save_vocab(self, path : str ="data/", filename : str ="vocabulary", format : str="pkl" ):

        vocab = {
            "word_to_index": self.word_to_index,
            "index_to_word": self.index_to_word,
            "negatives_distribution": self.negatives_distribution,
            "discard_probabilities": self.discard_probabilities
        }

        if format == "pkl":

            with open(f"{path}/{filename}.pkl", "wb") as f:
                pickle.dump(vocab, f)

        elif format == "json":

            with open(f"{path}/{filename}.json", "w") as f:
                json.dump(vocab, f, indent=2)

        else:
            raise ValueError("format must be 'pkl' or 'json'")


