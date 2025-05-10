# 导入必要的库
import cv2
import matplotlib as mpl

# 加载输入图像
# img = cv2.imread('res-2.png')
# img = cv2.imread('res-60.png')
# img = cv2.imread('res-100.png')
# img = cv2.imread('res-200.png')

# img = cv2.imread('res-apple.png')
# img = cv2.imread('res-apple-200.png')

# img = cv2.imread('res-blueberry.png')
# img = cv2.imread('res-blueberry-200.png')

# img = cv2.imread('res-cherry.png')
# img = cv2.imread('res-cherry-200.png')

# img = cv2.imread('i3.png')
img = cv2.imread('i3-93.png')

# 将图像转换为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用阈值处理以将灰度图像转换为二进制图像
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 找到轮廓
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("检测到的对象数量：", len(contours))


# 定义计算长宽比的函数
def aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    ratio = float(w) / h
    return ratio


# 选择第一个轮廓
cnt = contours[0]

# 找到长宽比
ar = aspect_ratio(cnt)

# 保留两位小数的长宽比
ar = round(ar, 3)

# 绘制轮廓
cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)

# 绘制边界矩形
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'无法显示的问题

# 在图像上放置文本
cv2.putText(img, f'长宽比={ar}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
print(f"对象1的长宽比=", ar)
cv2.imshow("长宽比", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
