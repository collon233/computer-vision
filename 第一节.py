# 帮我编写一个读取图片的函数，并对图片进行显示
# 1. 读取图片
# 2. 显示图片
# 3. 保存图片

# In[1]:

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_img(img):
    plt.imshow(img)
    plt.show()

def save_img(img, path):
    cv2.imwrite(path, img)
# 利用equalizeHist进行彩色图像直方图均衡化

def hist_equalize_equalizeHist_color(image):
    # 将彩色图像转换为YUV颜色空间
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 对Y通道进行直方图均衡化
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # 将图像转换回BGR颜色空间
    equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return equalized_image

def hist_equalize_equalizeHist_bio(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_equalized = cv2.equalizeHist(img)
    img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)
    return img_equalized

# 利用calcHist计算图像直方图


def hist_equalize_calcHist(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img_equalized


# 高斯滤波的函数
def gaussian_filter(img, kernel_size):
    dst = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return dst
# 中值滤波的函数
def median_filter(img, kernel_size):
    dst = cv2.medianBlur(img, kernel_size)
    return dst
# 双边滤波的函数
def bilateral_filter(img, kernel_size):

    dst = cv2.bilateralFilter(img, kernel_size, 75, 75)
    return dst
# FFT滤波的函数
def fft_filter(img, kernel_size):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, a, b = cv2.split(img_lab)
    dft_L = cv2.dft(np.float32(L), flags=cv2.DFT_COMPLEX_OUTPUT)    # 对灰度图像进行傅里叶变换
    dft_shift_L = np.fft.fftshift(dft_L)    # 将低频部分移动到左上角
    rows, cols = img.shape[:2]
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - kernel_size:crow + kernel_size, ccol - kernel_size:ccol + kernel_size] = 1  # 低通滤波器
    fshift = dft_shift_L * mask
    f_ishift = np.fft.ifftshift(fshift) # 将低频部分移动到左上角
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # 将img_back转化为uint8类型
    # img_back = cv2.convertScaleAbs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # 合并滤波后的L通道和原始的a、b通道
    img_filtered_lab = cv2.merge((img_back, a, b))

    # 将滤波后的Lab图像转换回RGB颜色空间
    img_filtered_rgb = cv2.cvtColor(img_filtered_lab, cv2.COLOR_Lab2RGB)

    return img_filtered_rgb

#均值滤波的函数
def mean_filter(img, kernel_size):
    dst = cv2.blur(img, (kernel_size, kernel_size))
    return dst

# def mean_filter(img, kernel_size):
#     # kernel_size: 3, 5, 7, 9, 11, 13, 15, 17, 19
#     # kernel_size = 3
#     # kernel_size = 5
#     # kernel_size = 7
#     # kernel_size = 9
#     # kernel_size = 11
#     # kernel_size = 13
#     # kernel_size = 15
#     # kernel_size = 17
#     # kernel_size = 19
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
#     img_filtered = cv2.filter2D(img, -1, kernel)
#     return img_filtered

# 我想把多张图利用plt合并在一起显示
def show_imgs(imgs, titles):
    for i in range(len(imgs)):
        plt.subplot(2, int(len(imgs)/2), i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main():
    img = read_img('../interlude_jk.png')

    hist_equalized_img = hist_equalize_calcHist(img)
    plt.plot(hist_equalized_img)
    img_equalized = hist_equalize_equalizeHist_color(img)
    hist_equalized_img_equalized = hist_equalize_calcHist(img_equalized)
    plt.plot(hist_equalized_img_equalized)
    plt.show()
    show_imgs([img, img_equalized], ['img', 'img_equalized'])

    # save_img(img, 'interlude_jk_test.png')


    img_filtered = mean_filter(img, 19)
    gaussian_filtered = gaussian_filter(img, 19)
    median_filtered = median_filter(img, 19)
    bilateral_filtered = bilateral_filter(img, 19)
    fft_filtered = fft_filter(img, 19)
    show_imgs([img, img_filtered, gaussian_filtered, median_filtered, bilateral_filtered, fft_filtered],
                ['img', 'img_filtered', 'gaussian_filtered', 'median_filtered', 'bilateral_filtered', 'fft_filtered'])



if __name__ == '__main__':
    main()


