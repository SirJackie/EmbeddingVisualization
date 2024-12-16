from gensim.models import KeyedVectors
import numpy as np


def PerspectiveProjection(vector):
    while vector.ndim == 1 and len(vector) > 3:
        n = len(vector)
        new_vector = vector[:n - 1] / vector[n - 1]
        vector = new_vector
    return vector


def Normalize(vector):
    norm = np.linalg.norm(vector)
    normalized_vector = vector / norm
    return normalized_vector


def Process(vector):
    return Normalize(PerspectiveProjection(vector))


print("Loading Model, Wait for 10s...")
model = KeyedVectors.load_word2vec_format("./Models/glove-wiki-gigaword-50.txt")

vec = model["tower"]
print(vec)
