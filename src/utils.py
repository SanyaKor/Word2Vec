import numpy as np

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return ( np.dot(vector_a, vector_b) ) / ((norm_a * norm_b) + 1e-10)