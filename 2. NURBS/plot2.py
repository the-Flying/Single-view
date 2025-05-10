import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成无序的三维点
# points = pd.read_excel('64_3.xlsx')
points = pd.read_excel('60_1.xlsx')
# points = pd.read_excel('B3-93.xlsx')
# points = pd.read_excel('evalpts_10000.xlsx')

# 将三维点分别存储为x、y、z数组
x = points.iloc[:, 0]
y = points.iloc[:, 1]
z = points.iloc[:, 2]

# 使用splprep函数拟合NURBS曲线
tck, u = splprep([x, y, z], s=0, per=True)

# 生成新的曲线点
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new, z_new = splev(u_new, tck)

# 绘制散点图和NURBS曲线
fig = plt.figure(figsize=(10.24, 8))
ax = fig.add_subplot(projection='3d')

# ax = Axes3D(fig)
fig.add_axes(ax)
for i in range(len(x)):
    ax.text(x[i], y[i], z[i], i)
    # print(x[i], y[i], z[i])

ax.scatter(x, y, z, color='red', label='Scatter Points')
ax.plot(x_new, y_new, z_new, color='blue', label='NURBS Contour')
ax.legend()
# 隐藏坐标轴
# ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0, 0)
plt.savefig('B1_2_NURBS.png', dpi=1000)
plt.show()
