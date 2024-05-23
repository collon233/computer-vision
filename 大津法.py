# 编写otsu方法
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 编写灰度直方图
def calculate_histogram(img):
    # 初始化灰度直方图
    hist = np.zeros([256], np.uint8)
    # 遍历图像的每个像素
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # 计算灰度直方图
            hist[img[row, col]] += 1
    return hist

def otsu(img):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 计算灰度直方图
    hist = calculate_histogram(gray_image)
    # 计算图像的总像素数
    total_pixel = gray_image.shape[0] * gray_image.shape[1]
    # 计算灰度直方图的概率分布
    hist = hist / total_pixel
    # 计算灰度直方图的均值
    mean = np.mean(gray_image)
    # 计算灰度直方图的方差
    variance = np.var(gray_image)
    # 初始化最大类间方差
    max_variance = 0
    # 初始化阈值
    threshold = 0
    # 遍历灰度直方图
    for i in range(256):
        # 计算类间方差
        variance = (mean * hist[i] - np.mean(gray_image[gray_image > i])) ** 2 / (hist[i] * (1 - hist[i]))
        # 判断类间方差是否大于最大类间方差
        if variance > max_variance:
            # 更新最大类间方差
            max_variance = variance
            # 更新阈值
            threshold = i
    # 对图像进行二值化处理
    ret, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    # 将图像转换为uint8类型
    otsu_image = cv2.convertScaleAbs(binary_image)
    print(threshold)
    return otsu_image

# otsu方法
def otsu_cv(img):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 对图像进行二值化处理
    ret, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 将图像转换为uint8类型
    otsu_image = cv2.convertScaleAbs(binary_image)
    return otsu_image


# 编写读取图像的方法
def read_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 将图像转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 编写显示多张图片的方法
def show_images(images, titles, rows, cols):
    # 设置图像的大小
    plt.figure(figsize=(16, 16))
    # 遍历图像列表
    for i in range(len(images)):
        # 绘制子图
        plt.subplot(rows, cols, i + 1)
        # 显示图像
        plt.imshow(images[i], 'gray')
        # 关闭坐标轴
        plt.axis('off')
        # 设置标题
        plt.title(titles[i])
    # 显示图像
    plt.show()

# 编写主函数
if __name__ == '__main__':
    # 读取图像
    image = read_image('yixue.png')
    # 图像大津法手动实现
    otsu_image = otsu(image)
    # 图像大津法opencv实现
    otsu_cv_image = otsu_cv(image)
    # 显示图像
    show_images([image, otsu_cv_image, otsu_cv_image], ['original image', 'otsu image', 'otsu cv image'], 1, 3)

