from src.vocabulary import Vocabulary

def test_vocab_build():

    tokens = ["cat", "dog", "cat", "mouse"]

    vocab = Vocabulary(min_words_count=1)
    vocab.build_vocab(tokens)

    assert "cat" in vocab.word_to_index
    assert "dog" in vocab.word_to_index
    assert vocab.vocab_size == 3