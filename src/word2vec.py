import numpy as np
import time
from .vocabulary import Vocabulary
import sys
import json
import os
import pickle

from .utils import cosine_similarity
from .corpus import Corpus

class Word2Vec:
    def __init__(self, embedding_size : int = 100, negatives_count : int = 5):

        self.embedding_size = embedding_size
        self.negatives_count = negatives_count

        self.vocabulary = None
        self.vocab_size = None

        self.word_embeddings = None
        self.context_embeddings = None

        self.eps = 1e-10
        self.lr = 0.01
        self.window_size = 5
        self.min_count = 2
        self.min_word_length = 2

    def train(self, training_samples : list[tuple[int, int, list[int]]], learning_rate : float = 0.001, epochs : int = 10):
        """
            Train the Word2Vec model using Skip-Gram with Negative Sampling.

            Iterates over precomputed training samples and performs SGD updates
            for each (center, context, negatives) tuple. Training progress is
            displayed in the console including loss, speed and ETA.

            After training completes the model (embeddings + vocabulary) is saved.
        """
        self.lr = learning_rate

        samples_count = len(training_samples)

        for epoch in range(epochs):

            np.random.shuffle(training_samples)

            total_energy = 0.0
            running_energy = 0.0

            start_time = time.time()

            for step, (word_index, context_index, negative_indexes) in enumerate(training_samples, start=1):

                energy = self._sgdl_step(
                    word_index,
                    context_index,
                    negative_indexes
                )

                total_energy += energy
                running_energy += energy


                ## NOTE** u can assign a smaller value, but it will significantly affect on speed
                if step % 1000 == 0:
                    elapsed = time.time() - start_time
                    avg_running_energy = running_energy / 1000
                    speed = step / elapsed if elapsed > 0 else 0.0
                    progress = step / samples_count * 100

                    remaining = samples_count - step
                    eta = remaining / speed if speed > 0 else 0.0
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))

                    sys.stdout.write(
                        f"\rEpoch {epoch + 1:>2}/{epochs:<2} | "
                        f"step {step:>7}/{samples_count:<7} | "
                        f"{progress:6.2f}% | "
                        f"loss: {avg_running_energy:8.4f} | "
                        f"{speed:7.0f} samples/s | "
                        f"ETA {eta_str}"
                    )
                    sys.stdout.flush()

                    running_energy = 0.0

            avg_energy = total_energy / samples_count
            epoch_time = time.time() - start_time
            epoch_speed = samples_count / epoch_time if epoch_time > 0 else 0.0

            sys.stdout.write(
                f"\rEpoch {epoch + 1:>2}/{epochs:<2} | "
                f"{100:6.2f}% | "
                f"loss: {avg_energy:8.4f} | "
                f"time: {epoch_time:7.2f}s | "
                f"samples: {samples_count:>8} | "
                f"speed: {epoch_speed:7.0f} samples/s\n"
            )
            sys.stdout.flush()

        self.save_model()

    def save_model(self, path : str = "data/embeddings"):

        os.makedirs(path, exist_ok=True)

        runs_path = os.path.join(path, "runs.json")

        if os.path.exists(runs_path) and os.path.getsize(runs_path) > 0:
            with open(runs_path, "r") as f:
                try:
                    runs = json.load(f)
                except json.JSONDecodeError:
                    runs = {}
        else:
            runs = {}

        run_id = 0
        while f"w2v_run{run_id}" in runs:
            run_id += 1

        run_name = f"w2v_run{run_id}"
        run_path = os.path.join(path, run_name)

        os.makedirs(run_path, exist_ok=True)

        np.savez(
            os.path.join(run_path, "embeddings.npz"),
            word_embeddings=self.word_embeddings,
            context_embeddings=self.context_embeddings
        )


        runs[run_name] = {
            "path": run_path,
            "embedding_size": self.embedding_size,
            "negatives_count": self.negatives_count,
            "window_size": self.window_size,
            "min_count": self.min_count,
            "min_word_length": self.min_word_length
        }

        self.vocabulary.save_vocab(os.path.join(run_path))

        with open(runs_path, "w") as f:
            json.dump(runs, f, indent=2)

    def load_model(self, path : str):
        """
            Load a trained Word2Vec model from disk.

            Restores the vocabulary mappings and sampling distributions from
            `vocabulary.pkl`, and loads the word and context embedding matrices
            from `embeddings.npz`.

            Args:
                path (str): Directory containing the saved model files.
        """
        with open(f"{path}/vocabulary.pkl", "rb") as f:
            vocab = pickle.load(f)

        self.vocabulary = Vocabulary(
            self.min_count,
            self.min_word_length
        )

        self.vocabulary.word_to_index = vocab["word_to_index"]
        self.vocabulary.index_to_word = vocab["index_to_word"]
        self.vocabulary.negatives_distribution = vocab["negatives_distribution"]
        self.vocabulary.discard_probabilities = vocab["discard_probabilities"]

        self.vocab_size = len(self.vocabulary.word_to_index)

        data = np.load(f"{path}/embeddings.npz", allow_pickle=True)

        self.word_embeddings = data["word_embeddings"]
        self.context_embeddings = data["context_embeddings"]

    def build_vocab(self, corpus : Corpus, min_count : int = 2, min_word_length : int = 2):

        self.min_count = min_count
        self.min_word_length = min_word_length

        raw_tokens = corpus.tokens

        self.vocabulary = Vocabulary(min_count, min_word_length)
        self.vocabulary.build_vocab(raw_tokens)

        token_ids = self.vocabulary.prepare_tokens(raw_tokens)

        self.vocab_size = len(self.vocabulary.word_to_index)

        self.word_embeddings = np.random.uniform(-0.8, 0.8,(self.vocab_size, self.embedding_size))
        self.context_embeddings = np.random.uniform(-0.8, 0.8,(self.vocab_size, self.embedding_size))

        return token_ids

    def most_similar(self, word : str, top_n : int = 5):

        if word not in self.vocabulary.word_to_index:
            raise ValueError(f"word '{word}' not in vocabulary")

        word_index = self.vocabulary.word_to_index[word]
        query_vector = self.word_embeddings[word_index]

        similarities = []

        for other_word, other_index in self.vocabulary.word_to_index.items():
            if other_word == word:
                continue

            other_vector = self.word_embeddings[other_index]
            similarity = cosine_similarity(query_vector, other_vector)

            similarities.append((other_word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]

    def build_training_samples(self, token_indexes : list[int], window_size : int = 5 ):
        """
           Builds training samples using the sliding window algorithm.

           For each center word, context words are collected from a window
           around it. For every (center, context) pair, negative samples are
           generated for training Skip-Gram with Negative Sampling.
        """

        print("Generating training pairs...")
        self.window_size = window_size
        training_samples = []

        token_count = len(token_indexes)

        for i, word_index in enumerate(token_indexes):

            if token_count > 0 and i % 1000 == 0:
                percent = (i / token_count) * 100
                sys.stdout.write(f"\rProgress: {percent:.1f}%")
                sys.stdout.flush()

            left = max(0, i - window_size)
            right = min(token_count, i + window_size + 1)

            for j in range(left, right):
                if i == j:
                    continue

                context_index = token_indexes[j]
                negative_indexes = self._sample_negatives(word_index, context_index)

                training_samples.append((word_index, context_index, negative_indexes))

        sys.stdout.write("\rProgress: 100.0%\n")
        print(f"Done. Generated {len(training_samples)} training samples.")
        return training_samples


    def _forward(self, word_index : int, context_index : int, negative_indexes : list[int]):
        """
            Performs the forward pass for a Skip-Gram with Negative Sampling (SGNS) sample.

            Computes the sigmoid scores for the positive (center, context) pair and the
            sampled negative pairs, then evaluates the energy (loss) for this training example.
        """

        word_vector = self.word_embeddings[word_index]

        context_vector = self.context_embeddings[context_index]
        negative_vectors = self.context_embeddings[negative_indexes]

        positive_output : float | np.ndarray = self._activation_function(np.dot(context_vector, word_vector))
        negative_output = self._activation_function(np.dot(negative_vectors, word_vector))

        energy = self._cost_function(positive_output, negative_output)

        return  positive_output, negative_output, energy

    def _backward(self, word_index : int , context_index : int , negative_indexes : list[int], positive_output : float, negative_output :  np.ndarray):
        """
            Performs the backward pass for a Skip-Gram with Negative Sampling (SGNS) sample.

            Computes gradients for the center word, the positive context word, and the
            sampled negative words, then updates their embeddings using SGD scheme.
        """

        # NOTE: word_vector, context_vector and negative_vectors were already
        # obtained in the forward pass. They are recomputed here to keep the
        # code simpler and avoid passing additional variables between methods.

        word_vector = self.word_embeddings[word_index]

        context_vector = self.context_embeddings[context_index]
        negative_vectors = self.context_embeddings[negative_indexes]

        context_update = (positive_output - 1.0) * word_vector
        negatives_update = np.outer(negative_output, word_vector)

        word_update = (positive_output - 1.0) * context_vector + np.dot(negative_output, negative_vectors)

        self.word_embeddings[word_index] -= self.lr * word_update
        self.context_embeddings[context_index] -= self.lr * context_update

        self.context_embeddings[negative_indexes] -= self.lr * negatives_update

    def _cost_function(self, positive_output : float, negative_output : np.ndarray ):
        """
            Computes the energy of a training sample under the negative sampling objective.
            Lower energy means the model assigns high probability to the positive pair
            and low probability to the negative samples.
            Note: `eps` is added for numerica stability to avoid log(0).
        """

        energy = -np.log(positive_output + self.eps) - np.sum(np.log(1.0 - negative_output + self.eps))
        return energy

    def _sample_negatives(self, word_index : int, context_index : int):
        """
            Samples negative word indices for Skip-Gram with Negative Sampling (SGNS).

            Draws words from the noise distribution (typically proportional to
            word frequency^0.75) while ensuring they are different from the
            center word and the positive context word.
        """
        negative_indices = [] ##

        while len(negative_indices) < self.negatives_count:
            idx = np.random.choice(self.vocab_size, p=self.vocabulary.negatives_distribution)

            if idx != word_index and idx != context_index:
                negative_indices.append(idx)

        return list(negative_indices)

    def _activation_function(self, x):
        """
            Sigmoid activation used in Skip-Gram with Negative Sampling (SGNS).

            Instead of computing a full softmax over the entire vocabulary,
            SGNS uses a binary objective (true context vs. negative samples).
            Sigmoid allows computing probabilities only for the positive pair
            and sampled negatives, avoiding evaluation of the full context matrix.

            Input may optionally be clipped to improve numerical stability.
        """
        x = np.asarray(x)
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def _sgdl_step(self, word_index : int , context_index : int, negative_indexes : list[int]):
        """
            Performs one SGD update for the Skip-Gram with Negative Sampling (SGNS) objective.

            Runs the forward pass to compute the outputs and energy (loss), then applies
            the backward pass to update the embeddings for the center word, context word,
            and negative samples.
        """

        positive_output, negative_output, energy = self._forward(word_index, context_index, negative_indexes)
        self._backward(word_index, context_index, negative_indexes, positive_output, negative_output)

        return energy


