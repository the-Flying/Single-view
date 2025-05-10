import cv2
import numpy as np
from matplotlib import pyplot as plt

# 第二张
img = cv2.imread("i1.png", flags=1)
height, width = img.shape[:2]  # 512 512
print(height, width)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV3
binary_inv = 255 - binary  # 反转颜色

fullCnts = np.zeros(img.shape[:2], np.uint8)  # 绘制轮廓函数会修改原始图像
fullCnts = cv2.drawContours(fullCnts, contours, -1, (255, 255, 255), thickness=0)  # 绘制全部轮廓，黑色

# 按轮廓的面积排序，绘制面积最大的轮廓
cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(1445, 1, 2)
maxCnt = np.zeros(img.shape[:2], np.uint8)  # 初始化最大轮廓图像
cv2.drawContours(maxCnt, cnts[0], -1, (0, 0, 0), thickness=3)  # 仅绘制最大轮廓 cnt，黑色
print("len(contours) =", len(contours))  # contours 所有轮廓的数量
print("area of max contour: ", cv2.contourArea(cnt))  # 轮廓面积
print("perimeter of max contour: {:.1f}".format(cv2.arcLength(cnt, True)))  # 轮廓周长

# 主成分分析方法提取目标的方向
markedCnt = maxCnt.copy()
ptsXY = np.squeeze(cnt).astype(np.float64)  # 删除维度为1的数组维度，(1445, 1, 2)->(1445, 2)
mean, eigenvectors, eigenvalues = cv2.PCACompute2(ptsXY, np.array([]))  # (1, 2) (2, 2) (2, 1)
print("mean:{}, eigenvalues:{}".format(mean.round(1), eigenvalues[:, 0].round(2)))

# 第三张
img2 = cv2.imread("i1-1.png", flags=1)
height2, width2 = img2.shape[:2]  # 512 512
print(height2, width2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, binary2 = cv2.threshold(gray2, 205, 255, cv2.THRESH_BINARY_INV)
binary2, contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV3
binary_inv2 = 255 - binary2  # 反转颜色

fullCnts2 = np.zeros(img2.shape[:2], np.uint8)  # 绘制轮廓函数会修改原始图像
fullCnts2 = cv2.drawContours(fullCnts2, contours2, -1, (255, 255, 255), thickness=0)  # 绘制全部轮廓，黑色

# 按轮廓的面积排序，绘制面积最大的轮廓
cnts2 = sorted(contours2, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
cnt2 = cnts2[0]  # 第 0 个轮廓，面积最大的轮廓，(1445, 1, 2)
maxCnt2 = np.zeros(img2.shape[:2], np.uint8)  # 初始化最大轮廓图像
cv2.drawContours(maxCnt2, cnts2[0], -1, (0, 0, 0), thickness=3)  # 仅绘制最大轮廓 cnt，黑色
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
plt.imshow(binary_inv, 'gray')  # 使用反转后的二值图像
plt.subplot(233), plt.axis('off'), plt.title("Contour4")
plt.imshow(255 - fullCnts, 'gray')  # 白色背景黑色轮廓

plt.subplot(234), plt.axis('off'), plt.title("Origin5")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot(235), plt.axis('off'), plt.title("Binary5")
plt.imshow(binary_inv2, 'gray')  # 使用反转后的二值图像
plt.subplot(236), plt.axis('off'), plt.title("Contour5")
plt.imshow(255 - fullCnts2, 'gray')  # 白色背景黑色轮廓

plt.tight_layout()
plt.show()

# # 保存图像
# fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
# ax = plt.Axes(fig, [0., 0., 1., 1.])  # 设置子图占满整个画布
# ax.set_axis_off()
# fig.add_axes(ax)
#
# # 第四张
# ax.imshow(fullCnts, cmap='gray')
# plt.savefig('FullCnts4.png')
#
# # 第五张
# ax.imshow(binary_inv, cmap='gray')  # 使用反转后的二值图像
# plt.savefig('Binary4.png')
#
# # 第六张
# ax.imshow(fullCnts2, cmap='gray')
# plt.savefig('FullCnts5.png')
#
# # 第七张
# ax.imshow(binary_inv2, cmap='gray')  # 使用反转后的二值图像
# plt.savefig('Binary5.png')

# 注意，这里加上plt.show()后，保存的图片就为空白了，因为plt.show()之后就会关掉画布，
# 所以如果要保存加显示图片的话一定要将plt.show()放在plt.savefig(save_path)之后
