import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation


# class MyAnnotation(Annotation):
#     def __init__(self, text, xyz, *args, **kwargs):
#         super().__init__(text, xy=(0, 0), *args, **kwargs)
#         self._xyz = xyz
#
#     def draw(self, renderer):
#         x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
#         self.xy = (x2, y2)
#         super().draw(renderer)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# data = pd.read_excel('64_4.xlsx')
# data = pd.read_excel('60_1.xlsx')
data = pd.read_excel('60-disorder.xlsx')
# data = pd.read_excel('cherry-order.xlsx')
x = data.iloc[:, 0]
y = data.iloc[:, 1]
z = data.iloc[:, 2]
print(len(x), len(y), len(z))

fig = plt.figure(figsize=(3.072, 3.072))
# fig.suptitle('3D Ordered Feature Points', fontweight='bold', fontsize=12)
ax = Axes3D(fig)
fig.add_axes(ax)
for i in range(len(x)):
    ax.text(x[i], y[i], z[i], i)
    print(x[i], y[i], z[i])
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'无法显示的问题
# ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], label='OFP', c="red")
ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c="red")
# ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], label='N F P', c="red")
# ax.plot(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], label='N F P', c="red")
# X、Y、Z坐标轴文本
# ax.set_xlabel('X', fontsize=16)
# # ax.set_xlim(0.0750, 0.0950)
# ax.set_ylabel('Y', fontsize=16)
# # ax.set_ylim(0.1325, 0.1525)
# ax.set_zlabel('Z', fontsize=16)
# # ax.set_zlim(-0.4825, -0.4625)
# # ax.legend(loc='upper right', fontsize=16, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
# #           ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
# # 绘制三维线框图像
# ax.plot(x, y, z, 'r.')
# 隐藏坐标轴
ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0, 0)
# plt.savefig('B3-93_point.png', dpi=1000)
plt.show()
