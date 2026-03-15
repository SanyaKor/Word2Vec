from .tokenizer import Tokenizer

class Corpus:

    def __init__(self, tokenizer=None, limit: int = 100_000_000):
        self.limit = limit
        self.tokenizer = tokenizer or Tokenizer()
        self.tokens = None

    def load_from_file(self, path, limit: int = 100_000_000):

        with open(path, "r") as f:
            text = f.read()

        tokens = self.tokenizer.tokenize(text)

        if limit:
            tokens = tokens[:limit]

        self.tokens = tokens

    def load_from_text(self, text, limit: int = 100_000_000):

        tokens = self.tokenizer.tokenize(text)

        if limit:
            tokens = tokens[:limit]

        self.tokens = tokens







