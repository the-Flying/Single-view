# 14.18 基于 PCA 的方向矫正 (OpenCV)
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 第二张
img = cv2.imread("i1.png", flags=1)
height, width = img.shape[:2]  # 512 512
print(height, width)
# src = cv2.resize(img, (300, 300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# print(retval, binary)
# 寻找二值化图中的轮廓，检索所有轮廓，输出轮廓的每个像素点
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # OpenCV4~
binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV3

fullCnts = np.zeros(img.shape[:2], np.uint8)  # 绘制轮廓函数会修改原始图像
fullCnts = cv2.drawContours(fullCnts, contours, -1, (255, 255, 255), thickness=0)  # 绘制全部轮廓

# 按轮廓的面积排序，绘制面积最大的轮廓
cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(1445, 1, 2)
maxCnt = np.zeros(img.shape[:2], np.uint8)  # 初始化最大轮廓图像
cv2.drawContours(maxCnt, cnts[0], -1, (255, 255, 255), thickness=3)  # 仅绘制最大轮廓 cnt
print("len(contours) =", len(contours))  # contours 所有轮廓的数量
print("area of max contour: ", cv2.contourArea(cnt))  # 轮廓面积
print("perimeter of max contour: {:.1f}".format(cv2.arcLength(cnt, True)))  # 轮廓周长

# 主成分分析方法提取目标的方向
markedCnt = maxCnt.copy()
ptsXY = np.squeeze(cnt).astype(np.float64)  # 删除维度为1的数组维度，(1445, 1, 2)->(1445, 2)
mean, eigenvectors, eigenvalues = cv2.PCACompute2(ptsXY, np.array([]))  # (1, 2) (2, 2) (2, 1)
print("mean:{}, eigenvalues:{}".format(mean.round(1), eigenvalues[:, 0].round(2)))

# # 绘制第一、第二主成分方向轴
# center = tuple(mean[0, :].astype(np.int64))  # 近似作为目标的中心 [266 281]
# e1xy = eigenvectors[0, :] * eigenvalues[0, 0]  # 第一主方向轴
# e2xy = eigenvectors[1, :] * eigenvalues[1, 0]  # 第二主方向轴
# p1 = (center + 0.01 * e1xy).astype(np.int64)  # P1:[149 403]
# p2 = (center + 0.01 * e2xy).astype(np.int64)  # P2:[320 332]
# theta = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi  # 第一主方向角度 133.6
# cv2.circle(markedCnt, center, 6, 255, -1)  # 在PCA中心位置画一个圆圈  RGB
# cv2.arrowedLine(markedCnt, center, p1, (255, 0, 0), thickness=3, tipLength=0.1)  # 从 center 指向 pt1
# cv2.arrowedLine(markedCnt, center, p2, (255, 0, 0), thickness=3, tipLength=0.2)  # 从 center 指向 pt2
# print("center:{}, P1:{}, P2:{}".format(center, p1, p2))
#
# # 根据主方向角度和中心旋转原始图像
# alignCnt = img.copy()
# cv2.circle(alignCnt, center, 8, (255, 255, 255), 2)  # 在PCA中心位置画一个圆圈  BGR
# cv2.arrowedLine(alignCnt, center, p1, (0, 0, 255), thickness=3, tipLength=0.1)  # 从 center 指向 pt1
# cv2.arrowedLine(alignCnt, center, p2, (0, 255, 0), thickness=3, tipLength=0.2)  # 从 center 指向 pt2
# x0, y0 = int(center[0]), int(center[1])
# print("x0={}, y0={}, theta={:.1f}(deg)".format(x0, y0, theta))
# MAR1 = cv2.getRotationMatrix2D((x0, y0), theta, 1)
# alignCnt = cv2.warpAffine(alignCnt, MAR1, alignCnt.shape[:2], borderValue=(255, 255, 255))  # 白色填充

# 第三张
img2 = cv2.imread("i1-1.png", flags=1)
height2, width2 = img2.shape[:2]  # 512 512
print(height2, width2)
# src = cv2.resize(img, (300, 300))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, binary2 = cv2.threshold(gray2, 205, 255, cv2.THRESH_BINARY_INV)

# 寻找二值化图中的轮廓，检索所有轮廓，输出轮廓的每个像素点
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # OpenCV4~
binary2, contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV3

fullCnts2 = np.zeros(img2.shape[:2], np.uint8)  # 绘制轮廓函数会修改原始图像
fullCnts2 = cv2.drawContours(fullCnts2, contours2, -1, (255, 255, 255), thickness=0)  # 绘制全部轮廓

# 按轮廓的面积排序，绘制面积最大的轮廓
cnts2 = sorted(contours2, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
cnt2 = cnts2[0]  # 第 0 个轮廓，面积最大的轮廓，(1445, 1, 2)
maxCnt2 = np.zeros(img2.shape[:2], np.uint8)  # 初始化最大轮廓图像
cv2.drawContours(maxCnt2, cnts2[0], -1, (255, 255, 255), thickness=3)  # 仅绘制最大轮廓 cnt
print("len(contours2) =", len(contours2))  # contours 所有轮廓的数量
print("area of max contour2: ", cv2.contourArea(cnt2))  # 轮廓面积
print("perimeter of max contour2: {:.1f}".format(cv2.arcLength(cnt2, True)))  # 轮廓周长

# 主成分分析方法提取目标的方向
markedCnt2 = maxCnt2.copy()
ptsXY2 = np.squeeze(cnt2).astype(np.float64)  # 删除维度为1的数组维度，(1445, 1, 2)->(1445, 2)
mean2, eigenvectors2, eigenvalues2 = cv2.PCACompute2(ptsXY2, np.array([]))  # (1, 2) (2, 2) (2, 1)
print("mean2:{}, eigenvalues2:{}".format(mean2.round(1), eigenvalues2[:, 0].round(2)))

# 显示图像
plt.figure(figsize=(6, 6))
plt.subplot(231), plt.axis('off'), plt.title("Origin4")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(232), plt.axis('off'), plt.title("Binary4")
plt.imshow(binary, 'gray')
plt.subplot(233), plt.axis('off'), plt.title("Contour4")
plt.imshow(fullCnts, 'gray')

plt.subplot(234), plt.axis('off'), plt.title("Origin5")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot(235), plt.axis('off'), plt.title("Binary5")
plt.imshow(binary2, 'gray')
plt.subplot(236), plt.axis('off'), plt.title("Contour5")
plt.imshow(fullCnts2, 'gray')

plt.tight_layout()
plt.show()

# 保存图像
fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
ax = plt.Axes(fig, [0., 0., 1., 1.])  # 设置子图占满整个画布
# 关掉x和y轴的显示
ax.set_axis_off()
fig.add_axes(ax)

# # 第四张
# ax.imshow(fullCnts, cmap='gray')
# plt.savefig('FullCnts4.png')
# plt.imshow(binary, 'gray')
# plt.savefig('Binary4.png')
#
# # 第五张
# ax.imshow(fullCnts2, cmap='gray')
# plt.savefig('FullCnts5.png')
# plt.imshow(binary2, 'gray')
# plt.savefig('Binary5.png')
#
# # 第六张
# ax.imshow(fullCnts3, cmap='gray')
# plt.savefig('FullCnts6.png')
# plt.imshow(binary3, 'gray')
# plt.savefig('Binary6.png')
'''
注意，这里加上plt.show()后，保存的图片就为空白了，因为plt.show()之后就会关掉画布，
所以如果要保存加显示图片的话一定要将plt.show()放在plt.savefig(save_path)之后
'''
# plt.show()
