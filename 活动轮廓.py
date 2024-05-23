import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """
    以参数方程的形式，获取n个离散点围成的圆形/椭圆形轮廓
    输入：中心centre=（x0, y0）, 半轴长radius=(a, b)， 离散点数N
    输出：由离散点坐标(x, y)组成的2xN矩阵
    """
    # 根据左上和右下的点绘制矩形

    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([y, x]).T

def plot(img):
    # 显示图像并交互式选择圆心
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap=plt.cm.gray)

    circle_center = []

    def on_click(event):
        if event.button == 1:  # 左键点击事件
            circle_center.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            plt.draw()

    plt.connect('button_press_event', on_click)
    plt.title('Click to select circle center')
    plt.axis('off')
    plt.show()

    # 提取圆心坐标
    if len(circle_center) > 0:
        center_x, center_y = circle_center[0]
        print(f"Selected circle center: ({center_x}, {center_y})")
    return center_x, center_y

img = data.astronaut()
# img = cv2.imread("../yixue.png")
cv2.imshow('img', img)
cv2.waitKey(0)
img = rgb2gray(img)
# img_resize = cv2.resize(img, (500,500))
#
# s = np.linspace(0, 2*np.pi, 400)
# r = 100 + 100*np.sin(s)
# c = 220 + 100*np.cos(s)
# init = np.array([r, c]).T

# 交互式选择左上和右下的点
pt = []
for i in range(2):
    pt.append(plot(img))
print(pt)
# 计算pt1和pt2之间的距离
dist = np.sqrt((pt[0][0] - pt[1][0]) ** 2 + (pt[0][1] - pt[1][1]) ** 2)

# 构造初始轮廓线
init = getCircleContour(pt[0], (dist,dist), N=400)
# print(init)

snake = active_contour(gaussian(img, 3),
                       init, alpha=0.01, beta=50, gamma=0.001)

fig, ax = plt.subplots(figsize=(8, 8))
pylab.imshow(img, cmap=pylab.gray())
pylab.plot(init[:,1], init[:,0], '--b', lw=3)
pylab.plot(snake[:, 1], snake[:, 0], '-r', lw=3)
pylab.axis('off')
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()