# 14.18 基于 PCA 的方向矫正 (OpenCV)
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("B6.jpg", flags=1)
height, width = img.shape[:2]  # 512 512
print(height, width)
src = cv2.resize(img, (300, 300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)

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
print("perimeter of max contour: {:.2f}".format(cv2.arcLength(cnt, True)))  # 轮廓周长

# 主成分分析方法提取目标的方向
markedCnt = maxCnt.copy()
# plt.imshow(markedCnt, 'gray')
# plt.show()
ptsXY = np.squeeze(cnt).astype(np.float64)  # 删除维度为1的数组维度，(1445, 1, 2)->(1445, 2)
mean, eigenvectors, eigenvalues = cv2.PCACompute2(ptsXY, np.array([]))  # (1, 2) (2, 2) (2, 1)
print("mean:{}, eigenvalues:{}".format(mean.round(1), eigenvalues[:, 0].round(2)))
# 绘制第一、第二主成分方向轴
center = tuple(mean[0, :].astype(np.int64))  # 近似作为目标的中心 [263 255]
# print(center)
e1xy = eigenvectors[0, :] * eigenvalues[0, 0]  # 第一主方向轴
e2xy = eigenvectors[1, :] * eigenvalues[1, 0]  # 第二主方向轴
p1 = tuple((center + 0.01 * e1xy).astype(np.int64))  # P1:[516 83]
# print(p1)
p2 = tuple((center + 0.01 * e2xy).astype(np.int64))  # P2:[422 491]
# print(p2)
theta = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi  # 第一主方向角度 -34.087056........
# print(theta)
cv2.circle(markedCnt, center, 6, 255, -1)  # 在PCA中心位置画一个圆圈  RGB
cv2.arrowedLine(markedCnt, center, p1, (255, 0, 0), thickness=3, tipLength=0.1)  # 从 center 指向 pt1
cv2.arrowedLine(markedCnt, center, p2, (255, 0, 0), thickness=3, tipLength=0.2)  # 从 center 指向 pt2
print("center:{}, P1:{}, P2:{}".format((263, 255), p1, p2))

# 根据主方向角度和中心旋转原始图像
alignCnt = img.copy()
cv2.circle(alignCnt, center, 8, (255, 255, 255), 2)  # 在PCA中心位置画一个圆圈  BGR
cv2.arrowedLine(alignCnt, center, p1, (0, 0, 255), thickness=3, tipLength=0.1)  # 从 center 指向 pt1
cv2.arrowedLine(alignCnt, center, p2, (0, 255, 0), thickness=3, tipLength=0.2)  # 从 center 指向 pt2
x0, y0 = int(center[0]), int(center[1])
print("x0={}, y0={}, theta={:.1f}(deg)".format(x0, y0, theta))
MAR1 = cv2.getRotationMatrix2D((x0, y0), theta, 1)
alignCnt = cv2.warpAffine(alignCnt, MAR1, alignCnt.shape[:2], borderValue=(255, 255, 255))  # 白色填充

# 显示图像
plt.figure(figsize=(9, 6))
plt.subplot(231), plt.axis('off'), plt.title("Origin")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(232), plt.axis('off'), plt.title("Binary")
plt.imshow(binary, 'gray')
plt.subplot(233), plt.axis('off'), plt.title("Contour")
plt.imshow(fullCnts, 'gray')
plt.subplot(234), plt.axis('off'), plt.title("Max contour")
plt.imshow(maxCnt, 'gray')
plt.subplot(235), plt.axis('off'), plt.title("Marked contour")
plt.imshow(markedCnt, 'gray')
plt.subplot(236), plt.axis('off'), plt.title("Alignment image")
plt.imshow(cv2.cvtColor(alignCnt, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
