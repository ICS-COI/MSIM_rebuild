import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BLUR_ONLY: int = 0
BLUR_GAUSS: int = 1
BLUR_POISSON: int = 2
BLUR_GP: int = 3

PAD_ZERO: int = 0
PAD_REFLECT: int = 1

INV_DIRECT: int = 0
INV_CUT: int = 1


def create_2d_gaussian_kernel(kernel_size, sigma):
    """
    创建二维高斯核

    :param kernel_size: 高斯核的大小，应为奇数
    :param sigma: 高斯分布的标准差
    :return: 二维高斯核数组
    """
    # 创建一个与高斯核大小相同的全零数组，中心位置设为1，代表单位脉冲（delta函数）
    kernel_shape = (kernel_size, kernel_size)
    impulse = np.zeros(kernel_shape)
    center = kernel_size // 2
    impulse[center, center] = 1

    # 对单位脉冲应用高斯滤波，得到高斯核
    gaussian_kernel = gaussian_filter(impulse, sigma)

    gaussian_kernel/=np.sum(gaussian_kernel)

    return gaussian_kernel


def list_suffix_files(folder_path, suffix):
    """
    读取指定文件夹中具有特定后缀的文件

    :param folder_path: 文件夹路径
    :param suffix: 特定后缀，如 '.txt'、'.jpg' 等
    :return: 符合条件的文件列表
    """
    specific_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                specific_files.append(os.path.join(root, file))

    return specific_files


def single_show(image, title):
    """
    该函数用于展示单张图像，并在图像标题处显示图像的形状信息。

    :param image: 要展示的图像数据，应为numpy数组格式
    :param title: 图像的标题字符串，会显示在图像上方，同时在标题下方添加图像的形状信息
    :return: None
    """
    plt.figure()
    if len(image.shape) == 3:
        plt.imshow(image[0], cmap='gray')

    elif len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    plt.title(title + '\n' + str(image.shape))
    plt.show()

    return


def blur_2d(img: np.ndarray, psf: np.ndarray, noise_flag: int, mean=0., sigma=0.001, pad=0, pad_flag=PAD_ZERO):
    """
    Image blur with edge extension (PSF + noise)

    :param img: Clear images
    :param psf: PSF
    :param noise_flag: Types of noise
        BLUR_ONLY - Noiseless
        BLUR_GAUSS - Gaussian noise
        BLUR_POISSON - Poisson noise
    :param mean: Gaussian noise mean
    :param sigma: Standard deviation of Gaussian noise
    :param pad: The number of edge-filled lines
    :param pad_flag: Padding mode
        PAD_ZERO - Zero padding
        PAD_REFLECT - Mirror padding
    :return: Blurred image
    """
    img_pad, psf_pad, padding = pad_2d(img, psf, pad, pad_flag)
    img_blur = convolution_2d(img_pad, psf_pad)
    if img_blur.max() != 0:
        img_blur /= img_blur.max() * 0.8
    img_blur = np.clip(img_blur, a_min=0, a_max=1)

    if noise_flag == BLUR_GAUSS:
        noise = np.random.normal(mean, sigma, img_pad.shape)
        img_blur = img_blur + noise
        # img_blur /= img_blur.max()
        img_blur = np.clip(img_blur, a_min=0, a_max=1)

    elif noise_flag == BLUR_POISSON:
        vals = len(np.unique(img_blur))
        vals = 2 ** np.ceil(np.log2(vals))
        # img_blur /= img_blur.max()
        img_blur = np.random.poisson(img_blur * vals) / np.float32(vals)

    elif noise_flag == BLUR_GP:
        vals = len(np.unique(img_blur))
        vals = 2 ** np.ceil(np.log2(vals))
        img_blur = np.random.poisson(img_blur * vals) / np.float32(vals)
        noise = np.random.normal(mean, sigma, img_pad.shape)
        img_blur = img_blur + noise
        img_blur /= img_blur.max()
        img_blur = np.clip(img_blur, a_min=0, a_max=1)

    if img_pad.shape != img.shape:
        return unpad_2d(img_blur, padding)
    return img_blur


def convolution_2d(img: np.ndarray, psf: np.ndarray):
    """
    3D convolution of clear images with PSF

    :param img: Clear images
    :param psf: PSF
    :return: A blurred 3D image after convolution
    """
    psf /= psf.max()
    if img.max() != 0:
        img /= img.max()
    otf = np.fft.fft2(psf)
    otf[np.where(np.abs(otf) < 1e-4)] = 0
    img_fft = np.fft.fft2(img)
    img_blur_fft = img_fft * otf
    img_blur = np.real(np.fft.fftshift(np.fft.ifft2(img_blur_fft)))
    return img_blur


def pad_2d(img: np.ndarray, psf: np.ndarray, pad, flag=PAD_REFLECT):
    """
    Pad a 3D image and it PSF for deconvolution

    :param img: Image array
    :param psf: Point Spread Function array
    :param pad: The number of edge-filled lines
    :param flag: Padding mode
    :return: image, psf, padding: padded versions of the image and the PSF, plus the padding tuple
    """
    padding = pad
    if isinstance(pad, int):
        if pad == 0:
            return img, psf, (0, 0)
        padding = (pad, pad)
    elif isinstance(pad, tuple):
        if len(pad) != img.ndim:
            raise Exception("Padding must be the same dimension as image")

    if padding[0] > 0 and padding[1] > 0:
        p2d = np.array([padding[0], padding[0], padding[1], padding[1]])
        if flag == PAD_REFLECT:
            img_pad = np.pad(img, p2d.reshape((2, 2)), "reflect")
        else:
            img_pad = np.pad(img, p2d.reshape((2, 2)), "constant")
        psf_pad = np.pad(psf, p2d.reshape((2, 2)), "constant")
    else:
        img_pad = img
        psf_pad = psf
    return img_pad, psf_pad, padding


def unpad_2d(img: np.ndarray, padding: tuple) -> np.ndarray:
    """
    Remove the padding of an image

    :param img: 3D image to unpad
    :param padding: Padding in each dimension
    :return: The unpadded image
    """
    return img[padding[0]:-padding[0], padding[1]:-padding[1]]


def save_tiff_3d(filename, img):
    img = np.uint16(img * 65535)
    stat = cv2.imwritemulti(filename, tuple(img),
                            (int(cv2.IMWRITE_TIFF_RESUNIT), 1, int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
    print(stat)
    if stat:
        print("Successfully save", filename, "!")
        return


def save_tiff_2d(filename, img):
    img = np.uint16(img * 65535)
    stat = cv2.imwrite(filename, img)
    if stat:
        print("Successfully save", filename, "!")
        return

def get_circular_region_coordinates_numpy(center_x, center_y, radius, image_shape):
    """
    使用numpy获取以(center_x, center_y)为中心，radius为半径的圆形区域内所有像素点坐标，
    并根据输入的图像宽度和高度对坐标进行限制，确保都在图像范围内
    :param center_x: 中心坐标的x值
    :param center_y: 中心坐标的y值
    :param radius: 圆形区域的半径
    :param image_shape: 图像形状
    :return: 圆形区域内像素点坐标列表，每个元素为一个二元组 (x, y)
    """
    # 确定横坐标（x）的有效范围，限制在图像宽度内
    x_min = max(center_x - radius, 0)
    x_max = min(center_x + radius, image_shape[0] - 1)
    x = np.arange(x_min, x_max + 1)

    # 确定纵坐标（y）的有效范围，限制在图像高度内
    y_min = max(center_y - radius, 0)
    y_max = min(center_y + radius, image_shape[1] - 1)
    y = np.arange(y_min, y_max + 1)

    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = distance <= radius
    coordinates = np.column_stack((xx[mask].ravel(), yy[mask].ravel())).tolist()
    return coordinates
