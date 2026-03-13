import re

class Tokenizer:
    def __init__(self, lower: bool = True, replace_numbers: bool = True):
        self.lower = lower
        self.replace_numbers = replace_numbers

    def tokenize(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()

        if self.replace_numbers:
            text = re.sub(r"\d+(?:\.\d+)?", " <num> ", text)

        text = re.sub(r"[_\-\/]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return re.findall(r"[a-z]+|<num>", text)