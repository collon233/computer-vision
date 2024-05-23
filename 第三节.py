

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 我想把多张图利用plt合并在一起显示
def show_imgs(imgs, titles, rows=2):
    #设置plt中字体可以显示中文
    plt.rcParams['font.family'] = 'Microsoft YaHei' # 设置字体
    for i in range(len(imgs)):
        if len(imgs) % 2 == 1:
            plt.subplot(rows, int(len(imgs)//rows)+1, i + 1)
        else:
            plt.subplot(rows, int(len(imgs)//rows), i + 1)
        plt.imshow(imgs[i], cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # ret, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO)

def Canny(img, threshold1, threshold2):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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

# Otsu函数编写


def kmeans_otsu(img,k = 2):
    # 获取图像的行数和列数
    rows, cols = img.shape[:2]
    # 将图像转换为二维数组
    data = img.reshape(rows * cols, -1).astype(np.float32)
    # 定义停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 迭代10次或者精度为1.0
    # 聚类
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 将聚类结果转换为图像
    center = np.uint8(center)
    max = np.max(center)
    min = np.min(center)
    # 将center中小的值变为0，大的值变为255
    center[center < min+1] = 0
    center[center > max-1] = 255
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    # print(set(label.flatten()))
    return result

# 编写计算图像的灰度均值的函数
def mean_gray(img):
    # 对不同灰度值的像素点进行计数
    gray_count = np.zeros(256)
    # 获取图像的行数和列数
    rows, cols = img.shape[:2]
    # 遍历图像的每个像素点
    for i in range(rows):
        for j in range(cols):
            gray_count[img[i, j]] += 1
    mean = np.mean(img)
    return mean

# 计算灰度直方图
def calculate_histogram(image):
    histogram = np.zeros(256, dtype=int)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1

    return histogram


# 直方图均衡化
def equalize_hist(img):
    # 直方图均衡化
    equalized_image = cv2.equalizeHist(img)
    # 将图像转换为uint8类型
    equalized_image = cv2.convertScaleAbs(equalized_image)
    return equalized_image

def gaussian(x, a, b, c):   #定义一个高斯函数
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def gaussian_mixture(hist):
    # 创建混合高斯模型对象
    n_components = 3  # 设定混合高斯模型中的组件数量
    model = GaussianMixture(n_components=n_components)

    # 拟合灰度直方图
    model.fit(hist.reshape(-1, 1))

    # 获取模型参数
    means = model.means_
    covariances = model.covariances_
    weights = model.weights_
    return means, covariances, weights,model

# 寻找直方图的极小值
def find_min(hist):
        minima = []
        length = len(hist)

        for i in range(1, length - 1):
            if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
                minima.append(i)

        return minima



# 提取图像最亮的区域
def bright_area(img):
    n_components = 3
    # 转换为灰度图像
    # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = np.arange(256)
    hist = calculate_histogram(img)
    # print(hist)
    # 绘制hist的灰度直方图
    plt.plot(hist)

    img = equalize_hist(img)
    hist = calculate_histogram(img)
    plt.plot(hist)
    plt.show()
    # 计算图像最大值
    max = np.max(img)

    #计算图像的灰度均值
    mean = np.mean(img)
    thread = (0.2 * mean + 0.8 * max)
    # 计算hist中均值的索引
    print(thread)
    img = otsu_hand(img,thread)
    return img


def yixue(img):
    # 转换为灰度图像
    k = 100
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.resize(gray_image, (k * 64, k * 64))
    combined_image = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)

    #将gray_image图像划分为4个区域，并保存为四个不同的图像
    for i in range(k):
        for j in range(k):
            sub_image = gray_image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64].copy()  # 提取当前区域的图像副本
            thresh = kmeans(sub_image)# 对当前区域应用阈值处理
            combined_image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = thresh  # 将处理后的图像添加到列表中
    #---------脑部分割----------------
    combined_image = bright_area(combined_image)
    #----------血管分割----------------
    # combined_image = median_filter(combined_image, 11)
    #
    # k = 2
    # combined_image = cv2.resize(combined_image, (k * 64, k * 64))
    # for i in range(k):
    #     for j in range(k):
    #         sub_image = combined_image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64].copy()
    #         thresh = kmeans_otsu(sub_image)
    #         combined_image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = thresh

    # Otsu算法
    ret, combined_image = cv2.threshold(combined_image, 0 ,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)#
    # 将图像转换为uint8类型

    otsu_image = cv2.convertScaleAbs(combined_image)
    return otsu_image

def median_filter(img, kernel_size):    # 中值滤波
    dst = cv2.medianBlur(img, kernel_size)
    return dst

# Otsu函数手动编写
def otsu_hand(img,threshold):
    # 转换为灰度图像
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 计算直方图和像素总数
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 初始化类间方差最大值和最佳阈值
    max_variance = 0
    best_threshold = threshold
    # # 遍历所有可能的阈值
    # for threshold in range(256):
    #     # 计算类别1和类别2的像素数和权重
    #     w1 = np.sum(hist[:threshold])
    #     w2 = np.sum(hist[threshold:])
    #     if w1 == 0 or w2 == 0:
    #         continue
    #     n1 = np.sum(hist[:threshold] * np.arange(threshold)) / w1  # 类别1的平均灰度
    #     n2 = np.sum(hist[threshold:] * np.arange(threshold, 256)) / w2 # 类别2的平均灰度
    #     # 计算类间方差
    #     variance = w1 * w2 * (n1 - n2) ** 2 #    类间方差 = 类别1权重 * 类别2权重 * (类别1平均灰度 - 类别2平均灰度) ** 2
    #     # 更新最大类间方差和最佳阈值
    #     if variance > max_variance:
    #         max_variance = variance
    #         best_threshold = threshold
    # 应用最佳阈值进行图像二值化
    binary = np.zeros_like(img)
    binary[img > best_threshold] = 255
    binary = cv2.convertScaleAbs(binary)

    return binary

# Kmeans对图像进行分割
def kmeans(img, k=2):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 获取图像的行数和列数
    rows, cols = img.shape[:2]
    # 将图像转换为二维数组
    data = img.reshape(rows*cols, -1).astype(np.float32)
    # 定义停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # 迭代10次或者精度为1.0
    # 聚类
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 将聚类结果转换为图像
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    print(set(label.flatten()))
    return result

# 高斯滤波函数
def gaussian_blur(img, ksize=3, sigma=1.5):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def hist_equalize_calcHist(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img_equalized

def hist_equalize_equalizeHist_bio(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_equalized = cv2.equalizeHist(img)
    img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)
    return img_equalized

def LoG(img, kernel_size):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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

if __name__ == '__main__':
    k = 2
    img = read_img('../yixue_01.png')
    img = gaussian_blur(img)
    img_otsu_cv = yixue(img)
    img_otsu_cv = cv2.resize(img_otsu_cv, (img.shape[1], img.shape[0]))
    output_image, canny_image = Canny(img_otsu_cv, 100, 200)

    kmeans_img = kmeans(img, k)
    #k = 2和大津法差不多
    show_imgs([img, img_otsu_cv, canny_image, kmeans_img], ['img', 'img_otsu_cv', 'canny_image', 'kmeans_img'])
    # 帮我编写一个保存图片的代码
    cv2.imwrite('img_otsu_cv.png', img_otsu_cv)


