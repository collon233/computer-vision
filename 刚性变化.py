import cv2
import numpy as np
# 读取图片
img = cv2.imread('tree.png')
height,width,channel = img.shape

# 利用2d仿射变化进行平移
def translate(image, x, y):

    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

# 利用2d进行仿射变化，center为旋转中心，scale为缩放大小
def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

# 利用2d仿射变化进行缩放变化
def suo_fang(img, fx,fy):
    M = np.float32([[fx, 0, 0], [0, fy, 0]])
    resized = cv2.warpAffine(img, M, (int(width*fx), int(height*fy)))

    return resized

def fan_zhuan(img,t = "垂直翻转"):
    if (t == "水平翻转"):
        M = np.float32([[-1, 0, width], [0, 1, 0]])
    elif(t == "垂直翻转"):
        M = np.float32([[1, 0, 0], [0, -1, height]])
    elif(t == "水平垂直翻转"):
        M = np.float32([[-1, 0, width], [0, -1, height]])
    flip =  cv2.warpAffine(img, M, (width, height))
    return flip

# 10,30 为向右和向上平移10个和30个单位的单位
shifted = translate(img, 10, 30)
# 90为角度
rotates = rotate(img,90)
# 3,1为x和y的缩放比例大小
resized = suo_fang(img,3,1)
# 利用2d仿射变化进行反转变化
for i in ["水平翻转","垂直翻转","水平垂直翻转"]:
    flip = fan_zhuan(img ,i)
    cv2.imwrite("flip_"+i+".jpg",flip)


cv2.imwrite('resize_raw.jpg', resized)
cv2.imwrite('shift_right_10_down_30.png', shifted)
cv2.imwrite('rotates_90_1.0.png', rotates)