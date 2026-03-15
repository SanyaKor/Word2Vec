import numpy as np
import time
from .vocabulary import Vocabulary
import sys
import json
import os
import pickle

from .utils import cosine_similarity

class Word2Vec:
    def __init__(self, vocab_size = None,  embedding_size : int = 100, negatives_count : int = 5):

        self.embedding_size = embedding_size
        self.negatives_count = negatives_count

        self.vocabulary = None
        self.vocab_size = vocab_size

        self.word_embeddings = None
        self.context_embeddings = None

        self.curr_lr = 0.01
        self.eps = 1e-10
        self.window_size = 5
        self.min_count = 2
        self.min_word_length = 2

        if vocab_size is not None:
            self._init_weights()


    def _init_weights(self):

        rnd = np.random.default_rng(42)

        #self.word_embeddings = np.random.randn(self.vocab_size, self.embedding_size)
        #self.context_embeddings = np.random.randn(self.vocab_size, self.embedding_size)

        ## BETTER initialization
        self.word_embeddings = rnd.uniform(-0.5 / self.embedding_size, 0.5 / self.embedding_size, (self.vocab_size, self.embedding_size))
        self.context_embeddings = rnd.uniform(-0.5 / self.embedding_size, 0.5 / self.embedding_size, (self.vocab_size, self.embedding_size))

    def train(self, training_samples : list[tuple[int, int, list[int]]], lr_start : float = 0.025, lr_end : float = 0.0001, epochs : int = 10):
        """
            Train the Word2Vec model using Skip-Gram with Negative Sampling.

            Iterates over precomputed training samples and performs SGD updates
            for each (center, context, negatives) tuple. Training progress is
            displayed in the console including loss, speed and ETA.

            After training completes the model (embeddings + vocabulary) is saved.
        """

        self.curr_lr = lr_start
        total_steps = epochs * len(training_samples)
        global_step = 0

        samples_count = len(training_samples)

        for epoch in range(epochs):

            np.random.shuffle(training_samples)

            total_energy = 0.0
            running_energy = 0.0

            start_time = time.time()

            for step, (word_index, context_index, negative_indexes) in enumerate(training_samples, start=1):

                self._update_learning_rate(global_step, total_steps, lr_start, lr_end)

                energy = self._sgdl_step(
                    word_index,
                    context_index,
                    negative_indexes
                )

                total_energy += energy
                running_energy += energy
                global_step += 1


            ### OUTPUT

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
                f"step {samples_count:>7}/{samples_count:<7} | "
                f"{100:6.2f}% | "
                f"loss: {avg_energy:8.4f} | "
                f"{epoch_speed:7.0f} samples/s | "
                f"time {epoch_time:05.2f}s\n"
            )
            sys.stdout.flush()

            ###

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

    def build_vocab(self, raw_tokens, min_count : int = 2, min_word_length : int = 1, subsample : bool = True):
        """
           Builds the vocabulary from the input tokens and prepares token ids for training.

           The method constructs a vocabulary based on word frequencies and filtering rules,
           optionally applies subsampling to reduce the impact of very frequent words,
           and converts the remaining tokens into their corresponding vocabulary indices.
           The resulting sequence of token ids can then be used to generate training
           samples for the model.
        """

        # NOTE:
        # `raw_tokens` refers to tokens produced directly by a tokenizer.
        # These tokens represent the segmented text but have not yet processed
        # any vocabulary filtering, subsampling, or index conversion

        self.min_count = min_count
        self.min_word_length = min_word_length

        self.vocabulary = Vocabulary(min_count)
        self.vocabulary.build(raw_tokens)
        if subsample:
            token_ids = [self.vocabulary.word_to_index[t] for t in self.vocabulary.subsample(raw_tokens) if t in self.vocabulary.word_to_index]
        else:
            token_ids = [self.vocabulary.word_to_index[t] for t in raw_tokens if t in self.vocabulary.word_to_index]

        self.vocab_size = len(self.vocabulary.word_to_index)

        self._init_weights()


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

        print("=" * 60 + "\n")

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
        print("\nTraining samples generated")
        print(f"Tokens processed   : {token_count}")
        print(f"Training samples   : {len(training_samples)}\n")
        print("=" * 60 + "\n")
        return training_samples

    def _update_learning_rate(self, global_step: int, total_steps: int, lr_start : float, lr_end : float):
        """
            Linearly decays the learning rate from lr_start to lr_end over training steps.
        """

        if total_steps <= 0:
            self.curr_lr = lr_end
            return

        progress = global_step / total_steps

        lr = lr_start * (1.0 - progress)

        self.curr_lr = max(lr, lr_end)

    def _forward(self, word_index : int, context_index : int, negative_indexes : list[int]):
        """
            Performs the forward pass for a Skip-Gram with Negative Sampling (SGNS) sample.

            Computes the scores from activation funcs for the positive (center, context) pair and the
            sampled negative pairs, then evaluates the energy (loss) for this training example.
        """

        word_vector = self.word_embeddings[word_index]

        context_vector = self.context_embeddings[context_index]
        negative_vectors = self.context_embeddings[negative_indexes]

        positive_output = self._activation_function(np.dot(context_vector, word_vector))
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

        self.word_embeddings[word_index] -= self.curr_lr * word_update

        self.context_embeddings[context_index] -= self.curr_lr * context_update
        self.context_embeddings[negative_indexes] -= self.curr_lr * negatives_update

    def _cost_function(self, positive_output : float | np.ndarray , negative_output : np.ndarray ):
        """
            Computes the energy of a training sample under the negative sampling objective.
            Lower energy means the model assigns high probability to the positive pair
            and low probability to the negative samples.

            Note: `eps` is added for numericall stability to avoid log(0).
        """

        energy = -np.log(positive_output + self.eps) - np.sum(np.log(1.0 - negative_output + self.eps))
        return energy

    def _sample_negatives(self, word_index : int, context_index : int):
        """
            Samples negative word indices for Skip-Gram with Negative Sampling (SGNS).

            Draws words from the noise distribution (typically proportional to
            word frequency^0.75 ) while ensuring they are different from the
            center word and the positive context word.
        """
        negative_indices = [] ##

        while len(negative_indices) < self.negatives_count:
            idx = np.random.choice(self.vocab_size, p=self.vocabulary.negatives_distribution)

            if idx != word_index and idx != context_index:
                negative_indices.append(idx)

        return list(negative_indices)

    def _activation_function(self, x: float | np.ndarray ) -> float | np.ndarray:
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
            Performs one Skip-Gram with Negative Sampling (SGNS) update using
            stochastic gradient descent (SGD) for a single training sample.

            Returns the sample energy (training loss).
        """

        positive_output, negative_output, energy = self._forward(word_index, context_index, negative_indexes)
        self._backward(word_index, context_index, negative_indexes, positive_output, negative_output)

        return energy


