from .tokenizer import Tokenizer

class Corpus:

    def __init__(self, path, tokenizer=None, limit: int = 100_000_000):
        self.path = path
        self.limit = limit
        self.tokenizer = tokenizer or Tokenizer()

        with open(path, "r") as file:
            text = file.read()

        tokens = self.tokenizer.tokenize(text)

        if self.limit:
            tokens = tokens[:self.limit]

        self._tokens = tokens

    def get_tokens(self):
        return self._tokens




