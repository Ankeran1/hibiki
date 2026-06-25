"""
文档扫描仪 - 二值化模块
支持 Otsu、自适应阈值、Sauvola 三种二值化方法
"""

import numpy as np
import cv2
from skimage.filters import threshold_sauvola


def binarize_otsu(img_gray: np.ndarray) -> np.ndarray:
    """Otsu 全局阈值二值化"""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(
    img_gray: np.ndarray,
    block_size: int = 25,
    C: int = 10,
) -> np.ndarray:
    """
    自适应高斯阈值二值化。
    block_size 必须为奇数。
    """
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C,
    )
    return binary


def binarize_sauvola(
    img_gray: np.ndarray,
    window_size: int = 35,
    k: float = 0.15,
) -> np.ndarray:
    """
    Sauvola 局部自适应阈值二值化。
    适用于光照不均、褪色或污损的文档。
    """
    if window_size % 2 == 0:
        window_size += 1
    thresh = threshold_sauvola(img_gray, window_size=window_size, k=k)
    binary = (img_gray > thresh).astype(np.uint8) * 255
    return binary


def binarize_auto(img_gray: np.ndarray) -> np.ndarray:
    """
    自动选择二值化方法。
    根据图像特征判断：
    - 检查光照均匀性：将图像分为4个象限，计算各象限均值差异
    - 光照均匀且高对比度 → Otsu
    - 光照不均或低对比度 → Sauvola
    """
    h, w = img_gray.shape[:2]
    # 分4象限检查光照均匀性
    quadrants = [
        img_gray[:h//2, :w//2],
        img_gray[:h//2, w//2:],
        img_gray[h//2:, :w//2],
        img_gray[h//2:, w//2:],
    ]
    means = [float(np.mean(q)) for q in quadrants]
    max_diff = max(means) - min(means)

    std = np.std(img_gray)

    # 光照差异大 或 对比度低 → Sauvola
    if max_diff > 30 or std <= 60:
        return binarize_sauvola(img_gray)
    else:
        return binarize_otsu(img_gray)


def enhance_scanner_output(img_gray: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    完整二值化增强流水线：
    1. 选择方法二值化
    2. 形态学开运算去除小噪点
    3. 返回干净的二值图像
    """
    # 二值化
    if method == "otsu":
        binary = binarize_otsu(img_gray)
    elif method == "adaptive":
        binary = binarize_adaptive(img_gray)
    elif method == "sauvola":
        binary = binarize_sauvola(img_gray)
    else:
        binary = binarize_auto(img_gray)

    # 形态学开运算去除孤立噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary
