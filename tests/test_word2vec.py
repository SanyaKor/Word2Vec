from src.corpus import Corpus
from src.vocabulary import Vocabulary
from src.word2vec import Word2Vec
import numpy as np


def test_vocab_build():

    tokens = ["cat", "dog", "cat", "mouse"]

    vocab = Vocabulary(min_words_count=1)
    vocab.build_vocab(tokens)

    assert "cat" in vocab.word_to_index
    assert "dog" in vocab.word_to_index
    assert vocab.vocab_size == 3


def test_train_training_samples():
    np.random.seed(0)
    model = Word2Vec(vocab_size=8 , embedding_size = 20, negatives_count = 5)

    training_samples = [
        (1, 2, [3,4,5]),
        (2, 3, [1,4,6]),
        (3, 4, [2,5,7]),
        (4, 5, [1,2,3]),
    ]

    energy_before = 0.0
    for word, ctx, neg in training_samples:
        p, n, energy = model._forward(word, ctx, neg)
        energy_before += energy

    for i in range(50):
        for word, ctx, neg in training_samples:
            model._sgdl_step(word, ctx, neg)

    energy_after = 0.0
    for word, ctx, neg in training_samples:
        p, n, energy = model._forward(word, ctx, neg)
        energy_after += energy


    assert energy_after < energy_before

def test_train_vocab():
    np.random.seed(0)
    text = """ While he was landing his Maurice Farman Shorthorn at the end of his first solo flight,
                another student collided with him and was killed, but Eaton emerged uninjured.
                He was commissioned in August and was awarded his wings in October.
                
                Ranked lieutenant, he served with No. 110 Squadron, which operated Martinsyde G.100
                "Elephant" fighters out of Sedgeford, defending London against Zeppelin airships.
                Transferred to the newly formed Royal Air Force (RAF) in April 1918,
                he was posted the following month to France flying Airco DH.9 single-engined
                bombers with No. 206 Squadron.
                
                On 29 June he was shot down behind enemy lines and captured in the vicinity
                of Nieppe. Incarcerated in Holzminden prisoner-of-war camp, Germany,
                Eaton escaped but was recaptured and court-martialled,
                after which he was kept in solitary confinement.
                
                He later effected another escape and succeeded in rejoining his squadron
                in the final days of the war.
                
                Between the wars Eaton remained in the RAF following the cessation
                of hostilities. He married Beatrice Godfrey in St. Thomas's Church
                at Shepherd's Bush, London, on 11 January 1919.
                
                Posted to No. 1 Squadron, he was a pilot on the first regular passenger
                service between London and Paris, ferrying delegates to and from
                the Peace Conference at Versailles.
                
                Eaton was sent to India in December to undertake aerial survey work,
                including the first such survey of the Himalayas. He resigned from
                the RAF in July 1920 but remained in India to work with the Imperial
                Forest Service.
                
                After successfully applying for a position with the Queensland Forestry
                Service, he and his family migrated to Australia in 1923.
                Moving to South Yarra, Victoria, he enlisted as a flying officer
                in the Royal Australian Air Force (RAAF) at Laverton on 14 August 1925.
                
                He was posted to No. 1 Flying Training School at RAAF Point Cook
                as a flight instructor, where he became known as a strict disciplinarian
                who "trained his pilots well".
        """

    corpus = Corpus()
    corpus.load_from_text(text)

    w2v = Word2Vec()


    encoded_corpus = w2v.build_vocab(corpus, subsample=False)
    training_samples = w2v.build_training_samples(encoded_corpus)

    energy_before = 0.0
    for word_index, context_index, negative_indexes in training_samples:
        p, n, energy = w2v._forward(word_index, context_index, negative_indexes)
        energy_before += energy

    w2v.train(training_samples)

    energy_after = 0.0
    for word_index, context_index, negative_indexes in training_samples:
        p, n, energy = w2v._forward(word_index, context_index, negative_indexes)
        energy_after += energy

    assert energy_after < energy_before


def test_most_similar():
    np.random.seed(0)
    text = """
            the king lived in the castle and the queen lived in the castle
            the king spoke to the queen in the great hall
            the queen answered the king in the great hall
            the king and the queen walked through the royal garden
            the king and the queen rode together across the quiet valley
            
            the man worked in the village and the woman worked in the village
            the man spoke to the woman near the river
            the woman answered the man near the river
            the man and the woman walked through the market
            the man and the woman rode together across the old bridge
            
            the prince learned from the king and the princess learned from the queen
            the prince followed the king into the castle
            the princess followed the queen into the castle
            the prince greeted the princess in the garden
            the princess greeted the prince in the garden
            
            paris is a city in france and berlin is a city in germany
            the traveler went from paris to france by train
            the traveler went from berlin to germany by train
            paris and france were spoken of together in the story
            berlin and germany were spoken of together in the story
            
            the king loved the queen and the queen trusted the king
            the man loved the woman and the woman trusted the man
            the prince admired the princess and the princess admired the prince
            
            one morning the king met the queen in the garden
            one morning the man met the woman in the market
            one morning the prince met the princess near the castle
            
            the king returned to the castle with the queen
            the man returned to the village with the woman
            the prince returned to the palace with the princess
        """

    corpus = Corpus()
    corpus.load_from_text(text)

    w2v = Word2Vec(embedding_size=20, negatives_count=2)

    encoded = w2v.build_vocab(corpus, subsample=False)
    samples = w2v.build_training_samples(encoded)

    w2v.train(samples, epochs=20)

    similar = w2v.most_similar("king", top_n=3)

    words = [w for w, _ in similar]

    assert "queen" in words








