
class Corpus:

    def __init__(self, path, limit=200_000, sentence_length=1000):
        self.path = path
        self.limit = limit
        self.sentence_length = sentence_length

        with open(path, "r") as file:
            text = file.read()

        self.tokens = text.split()

