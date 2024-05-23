

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature as ft

def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#将图像转换为灰度图像
def gray_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def show_img(img):
    plt.imshow(img)
    plt.show()
#
# #利用svm进行数字识别，并svm的描述符为hog
# def svm_hog(img):
#
#
#
#
#
#
# #帮我写个利用hog检测行人的函数
# def hog_detect(img):
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#     (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
#     return rects, weights

# hog检测图像特征的函数
def hog(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)  # 计算hog特征 3780维   3780 = 64*7*7
    features = ft.hog(img, orientations=6, pixels_per_cell=[20, 20], cells_per_block=[2, 2], visualize=True)
    return h, features

# LoG算子的函数
def LoG(img, kernel_size):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯平滑
    gaussian_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

    # 将图像转换为float32类型
    # gray_float32 = np.float32(gaussian_image)

    # 计算拉普拉斯算子
    laplacian_image_log = cv2.Laplacian(gaussian_image, cv2.CV_64F)
    laplacian_image_bio = cv2.Laplacian(gray_image, cv2.CV_64F)

    laplacian_image_img = laplacian_image_log.astype(img.dtype) # 转换回原来的数据类型
    laplacian_image_color = laplacian_image_bio.astype(img.dtype) # 转换回原来的数据类型

    gray_to_color = cv2.cvtColor(laplacian_image_img, cv2.COLOR_GRAY2BGR)
    laplacian_image_color = cv2.cvtColor(laplacian_image_color, cv2.COLOR_GRAY2BGR)

    # 合并原始彩色图像和变换后的灰度图像
    output_image = cv2.addWeighted(img, 0.7, gray_to_color, 0.3, 0)
    laplacian_image_color = cv2.addWeighted(img, 0.7, laplacian_image_color, 0.3, 0)
    # 将图像转换为uint8类型
    output_image =  cv2.convertScaleAbs(output_image)
    laplacian_image_log =  cv2.convertScaleAbs(laplacian_image_log)
    laplacian_image_demo =  cv2.convertScaleAbs(laplacian_image_bio)
    laplacian_image_color =  cv2.convertScaleAbs(laplacian_image_color)
    return output_image, laplacian_image_log, laplacian_image_demo, laplacian_image_color

# Canny算子的函数
def Canny(img, threshold1, threshold2):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 计算Canny算子
    canny_image = cv2.Canny(gray_image, threshold1, threshold2)
    canny_image_img = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
    canny_image_img = canny_image_img.astype(img.dtype)

    # 将图像转换为uint8类型
    canny_image = cv2.convertScaleAbs(canny_image)

    # 合并原始彩色图像和变换后的灰度图像
    output_image = cv2.addWeighted(img, 0.7, canny_image_img, 0.3, 0)

    return output_image, canny_image

# hough变换的函数
def Hough(img):

    # 霍夫变换
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # 绘制直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    show_img(img)

#减少图像的尺寸
def resize_img(img, size):
    img = cv2.resize(img, size)
    return img

# 我想把多张图利用plt合并在一起显示
def show_imgs(imgs, titles, rows=2):
    #设置plt中字体可以显示中文
    plt.rcParams['font.family'] = 'Microsoft YaHei' # 设置字体
    for i in range(len(imgs)):
        plt.subplot(rows, int(len(imgs)//rows), i + 1)
        plt.imshow(imgs[i], cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 写个启动函数
def main():
    img = read_img('../interlude_jk.png')

    img_gray = gray_img(img)
    h,features = hog(img_gray)
    print(h.shape)

    show_imgs([img,features[1]], ['原始图像','hog的像素特征方向和大小'])
    img_color,img_bio,img_l_demo_bio,img_l_demo_color = LoG(img, 15)
    canny_color,canny_bio = Canny(img, 100, 200)

    show_imgs([img_color,img_bio,img_l_demo_color,img_l_demo_bio,canny_color,canny_bio],
              ['LoG彩色图片', 'LoG灰度图片',"Laplacian彩色图片","Laplacian黑白图片","Canny算子彩色图片","Canny算子黑白图片"],
              rows=3)

    Hough(canny_bio)

if __name__ == '__main__':
    main()

