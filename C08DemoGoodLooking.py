from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


def SaveVec(var, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var, f)


def LoadVec(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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

    # Input Words
    w1 = "hitler"
    w2 = "germany"
    w3 = "italy"

    # Calculate Vectors
    v1 = LoadVec("v1.pkl")
    v2 = LoadVec("v2.pkl")
    v3 = LoadVec("v3.pkl")

    v4 = v1 - v2 + v3
    closest_word = "mussolini"
    v4_closest = LoadVec("v4_closest.pkl")
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

    v4p = v1p - v2p + v3p
    difference_visual = (v1p + v2p + v3p) * 0.05
    v4cp = v1p - v2p + v3p + difference_visual

    # 绘制向量
    origin = np.array([0, 0, 0])
    ax.quiver(*origin, *v1p, color='r', arrow_length_ratio=0.1, label=w1)
    ax.quiver(*v1p, *(-v2p), color='g', arrow_length_ratio=0.1, label=w2 + " * -1")
    ax.quiver(*(v1p-v2p), *v3p, color='b', arrow_length_ratio=0.1, label=w3)
    ax.quiver(*origin, *v4p, color='purple', arrow_length_ratio=0.1, label="IDEAL_RESULT")
    ax.quiver(*origin, *v4cp, color='aqua', arrow_length_ratio=0.1, label=closest_word)

    # 设置坐标轴范围和标签
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # 显示图形
    plt.show()

    pass
