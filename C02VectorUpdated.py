import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 定义四个向量，构成平行四边形
v1 = np.array([1, 2, 3])
v2 = np.array([3, 1, 1])
v3 = v1 + v2
# v4 = -v1 + v2

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制向量
origin = np.array([0, 0, 0])
ax.quiver(*origin, *v1, color='r', arrow_length_ratio=0.1, label="v1")
ax.quiver(*origin, *v2, color='g', arrow_length_ratio=0.1, label="v2")
ax.quiver(*v1, *v2, color='b', arrow_length_ratio=0.1, label="v1+v2")
# ax.quiver(*origin, *v4, color='purple', arrow_length_ratio=0.1, label="-v1+v2")

# 设置坐标轴范围和标签
ax.set_xlim([-1, 5])
ax.set_ylim([-1, 5])
ax.set_zlim([-1, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# 显示图形
plt.show()
