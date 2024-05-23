import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature as ft
# 读取图片
img_demo = cv2.imread('interlude_jk.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img_demo)
plt.imshow(img_demo,cmap=plt.cm.gray)
plt.show()
# hog---------------------
features = ft.hog(img_demo,orientations=6,pixels_per_cell=[20,20],cells_per_block=[2,2],visualize=True)
plt.imshow(features[1],cmap=plt.cm.gray)
plt.show()
#
# # SIFT-----------------------------------
img2 = cv2.cvtColor(cv2.imread('interlude_jk_test.png'), cv2.COLOR_BGR2GRAY)
img1 = img_demo
#
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)

matches = sorted(matches, key=lambda  x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:20], img2, flags=2)
plt.imshow(img3)
plt.show()

x = cv2.Sobel(img1, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img1, cv2.CV_16S, 0, 1)
Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
Scale_absY = cv2.convertScaleAbs(y)
result_sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

# Roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(img1, cv2.CV_16S, kernelx)
y = cv2.filter2D(img1, cv2.CV_16S, kernely)
# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(img1, cv2.CV_16S, kernelx)
y = cv2.filter2D(img1, cv2.CV_16S, kernely)
# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# canny算子
edges = cv2.Canny(img1, 100, 200)

# 显示图形
plt.imshow(Prewitt, cmap=plt.cm.gray), plt.title('Prewitt算子'), plt.axis('off')
plt.show()
plt.imshow(Roberts, cmap=plt.cm.gray), plt.title('Roberts算子'), plt.axis('off')
plt.show()


# ----------------------显示结果----------------------------
cv2.imwrite('sobel_Scale_absX.jpg', Scale_absX)
cv2.imwrite('sobel_Scale_absY.jpg', Scale_absY)
cv2.imwrite('sobel_result.jpg', result_sobel)
cv2.imwrite('canny_result.jpg', edges)
