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
        self.filtered_word_count = 0
        self.vocab_size = 0

        ### dict for O(1) operations word -> idx
        self.word_to_index = {}

        ### list for O(1) operations idx -> word
        self.index_to_word = []

        self.negatives_distribution = None
        self.discard_probabilities = None


    def build(self, tokens: list[str]):
        """
        Build the vocabulary from input tokens.

        Computes token frequencies, filters words by minimum frequency and
        minimum word length, assigns indices, and constructs the negative
        sampling distribution.
        """

        self.word_freqs_map = dict(Counter(tokens))

        print("\nBuilding vocabulary...")
        print(f"Total tokens         : {len(tokens)}")

        filtered_words = {}

        for word, count in self.word_freqs_map.items():
            if count >= self.min_words_count and len(word) >= self.min_word_length:
                filtered_words[word] = count

        sort_filtered_words = sorted(filtered_words.items(), key=lambda item: item[1],reverse=True)

        for index, (word, count) in enumerate(sort_filtered_words):
            self.word_to_index[word] = index
            self.index_to_word.append(word)

        word_freqs_list = []

        for word in self.index_to_word:
            count = filtered_words[word]
            word_freqs_list.append(count)


        self.vocab_size = len(self.index_to_word)

        print(f"Filtered vocabulary  : {self.vocab_size}\n")

        self._build_negatives_distribution(filtered_words)
        self._build_discard_probabilities(filtered_words)


    def _build_discard_probabilities(self, words_map: dict[str, int]):
        """
        Build discard probabilities for subsampling frequent words.
        """
        total_count = sum(words_map.values())

        self.discard_probabilities = {}

        for word in self.index_to_word:
            count = words_map[word]
            freq = count / total_count

            discard_probability = 1.0 - np.sqrt(self.const_t / freq)
            discard_probability = np.clip(discard_probability, 0.0, 1.0)

            self.discard_probabilities[word] = float(discard_probability)


    def _build_negatives_distribution(self, words_map: dict[str, int]):

        self.word_freqs = []
        freq_pow = []
        total = 0.0

        for word in self.index_to_word:
            freq = words_map[word]
            self.word_freqs.append(freq)

            value = freq ** 0.75
            freq_pow.append(value)
            total += value

        self.word_freqs = np.array(self.word_freqs)
        self.negatives_distribution = np.array(freq_pow) / total

    def subsample(self, tokens: list[str]) -> list[str]:
        """
        Apply subsampling to tokens.

        Frequent words are probabilistically discarded according to the
        subsampling distribution. Tokens not present in the vocabulary
        are kept unchanged.
        """

        res = []

        for token in tokens:
            prob = self.discard_probabilities.get(token)

            if prob is None:
                res.append(token)
                continue

            if np.random.rand() >= prob:
                res.append(token)

        return res




    def prepare_tokens(self, tokens : list[str], subsample : bool = True):
        """
          Applies subsampling to the input tokens and converts the remaining tokens
          to their corresponding vocabulary indices.

          Tokens not present in the vocabulary are skipped. The resulting list of
          token ids is ready to be used for training.
        """

        # NOTE: Combines token subsampling and token-to-id conversion in one pass.
        # This is faster than running the two steps separately, which helps reduce
        # preprocessing time before building training pairs.


        tokens_id = []

        for word in tokens:
            if word not in self.word_to_index:
                continue

            if subsample:
                if np.random.rand() >= self.discard_probabilities[word]:
                    tokens_id.append(self.word_to_index[word])
            else:
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


