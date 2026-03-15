from src.corpus import Corpus
from src.word2vec import Word2Vec
import numpy as np

np.random.seed(42)


text = """
the king lived in the castle and the queen lived in the castle
the king spoke to the queen in the great hall
the queen answered the king in the great hall
the king and the queen walked through the royal garden
the king and the queen rode together across the quiet valley
the man worked in the village and the woman worked in the village
the man spoke to the woman near the river
the woman answered the man near the river
"""

corpus = Corpus()
corpus.load_from_text(text)
print(corpus.tokens)


w2v = Word2Vec(embedding_size=50, negatives_count=2)
tok_ids = w2v.build_vocab(corpus.tokens, subsample=False, min_count=1)

samples = w2v.build_training_samples(tok_ids)
w2v.train(samples, epochs=100)
print(w2v.most_similar("king", top_n=5))