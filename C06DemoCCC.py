from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


if __name__ == "__main__":

    #
    # Calculate Embedding
    #

    # Load the Embedding Model
    print("Loading Model, Wait for 10s...")
    model = KeyedVectors.load_word2vec_format("./Models/glove-wiki-gigaword-50.txt")

    # Input Words
    # w1 = input("Word 1: ")
    # w2 = input("Word 2: ")
    # w3 = input("Word 3: ")
    w1 = "hitler"
    w2 = "germany"
    w3 = "italy"

    # Calculate Vectors
    v1 = model[w1]
    v2 = model[w2]
    v3 = model[w3]

    v4 = v1 - v2 + v3
    closest_word, closest_probability = model.similar_by_vector(v4, topn=1)[0]
    v4_closest = model[closest_word]
    print(f"----- RESULT: {w1} - {w2} + {w3} = {closest_word} -----")

    #
    # Visualize
    #

    # Create 3D Axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualize
    v1p = Process(v1)
    v2p = Process(v2)
    v3p = Process(v3)
    v4p = Process(v4)
    v4cp = Process(v4_closest)

    # 绘制向量
    origin = np.array([0, 0, 0])
    ax.quiver(*origin, *v1p, color='r', arrow_length_ratio=0.1, label="v1")
    ax.quiver(*v1p, *(-v2p), color='g', arrow_length_ratio=0.1, label="-v2")
    ax.quiver(*(-v2p), *v3p, color='b', arrow_length_ratio=0.1, label="v3")
    # ax.quiver(*origin, *v4p, color='purple', arrow_length_ratio=0.1, label="-v1+v2")

    # 设置坐标轴范围和标签
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # 显示图形
    plt.show()

    pass
