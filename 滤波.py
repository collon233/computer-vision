import cv2
import numpy as np
#读取图片
img_name = '01.png'
img = cv2.imread(img_name)
#设置高斯分布的均值和方差
mean = 0
#设置高斯分布的标准差
sigma = 0.05

# 设置添加椒盐噪声的数目比例
s_vs_p = 0.5
# 设置添加噪声图像像素的数目
amount = 0.1
def gasuss_noise(img,m,s):
    image = np.array(img/255,dtype=float)
    gauss = np.random.normal(m,s,image.shape)
    noisy_img = image + gauss
    if noisy_img.min()<0:
        low_clip=-1.
    else:
        low_clip = 0.

    noisy_img=np.clip(noisy_img,low_clip,1.0)
    noisy_img=np.uint8(noisy_img*255)
    return noisy_img

def sp_noise(image,a,sp):
    noisy_img = np.copy(image)
    num_salt = np.ceil(a * image.size * sp)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords] = 255
    num_pepper = np.ceil(a * image.size * (1. - sp))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords] = 0
    return noisy_img

def GHPF(img,):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    f = fshift * mask

    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    return res

def GLPF(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器

    rows, cols = img.shape

    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.zeros((rows, cols, 2), np.uint8)

    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 掩膜图像和频谱图像乘积

    f = fshift * mask

    # 傅里叶逆变换

    ishift = np.fft.ifftshift(f)

    iimg = cv2.idft(ishift)

    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    return res

noisy_img_gasuss = gasuss_noise(img,mean,sigma)
noisy_img_sp = sp_noise(img,amount,s_vs_p)

gasuss_img = cv2.GaussianBlur(img, (11, 11), 5)
box_img = cv2.blur(img, (7, 7))

gasuss_img_filer = cv2.GaussianBlur(noisy_img_gasuss,(7,7),2)
img_mid = cv2.medianBlur(noisy_img_gasuss, 11)
img_GHPF = GHPF(noisy_img_gasuss)

cv2.imwrite(img_name+'_gasuss.jpg', gasuss_img)
cv2.imwrite(img_name+'_box.jpg', box_img)
cv2.imwrite(img_name+'_sp_noisy.jpg', noisy_img_sp)
cv2.imwrite(img_name+'_gasuss_noisy.jpg', noisy_img_gasuss)
cv2.imwrite(img_name+'_gasuss_img_filer.jpg', gasuss_img_filer)
cv2.imwrite(img_name+'img_GHPF.jpg', img_GHPF)
cv2.imwrite(img_name+'img_mid.jpg', img_mid)







# # -------------------------------------------------------------------------------------------------------------
# gray=cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
# # 2340 1080
# N=gray.size
# print(N)
# print(gray.shape)
# pro=np.array([0]*256)
# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         pro[gray[i,j]]+=1
# pro=pro/N
#
# T=1 #0
# delta=0
# thresh=0
# while T<=256:
#     w0=pro[0:T].sum()
#     w1=pro[T+1:256].sum()
#     u0=(pro[0:T]*range(1,T+1)).sum()/w0
#     u1=(pro[T:256]*range(T+1,257)).sum()/w1
#     u=w0*u0+w1*u1
#     v=w0*w1*np.square(u1-u0)
#     if v>delta:
#         delta=v
#         thresh=T
#     T+=1
# thresh=100
# otsu=np.zeros(gray.shape,np.uint8)
# for i in range(gray.shape[0]):
#     for j in range(gray.shape[1]):
#         if gray[i,j]<thresh:
#             otsu[i,j]=255
# # cv2.contourArea()
# print(thresh)
#
# # edges = cv2.Canny(gray, 64, 256)
# # cv2.imshow('Otsu',edges)
# # cv2.imwrite(img_name+"_edges.jpg",edges)
# cv2.imwrite(img_name+"_otsu.jpg",otsu)