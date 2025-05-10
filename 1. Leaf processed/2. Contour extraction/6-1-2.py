# 导入必要的库
import cv2

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


# 定义计算范围的函数
def extent(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = float(area) / rect_area
    return extent


# 选择第一个轮廓
cnt = contours[0]

# 查找范围
ext = extent(cnt)

# 将范围四舍五入到三个小数点
ext = round(ext, 6)

# 绘制轮廓
cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)

# 绘制边框矩形
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 在图像上放置文本
cv2.putText(img, f'Extent={ext}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

print("对象的范围：", ext)
cv2.imshow("范围", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
