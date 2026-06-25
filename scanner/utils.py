"""
文档扫描仪 - 工具函数模块
图像加载/保存、缩放、坐标变换等基础操作
"""

import os
import numpy as np
import cv2

# ── 常量 ──────────────────────────────────────────────
MAX_DETECT_SIZE: int = 1024           # 检测时最大边长（兼顾速度与精度）
WARP_PADDING: int = 20                # 透视矫正输出边缘留白
MIN_CONTOUR_AREA_RATIO: float = 0.01  # 轮廓面积最小占比
MIN_SOLIDITY: float = 0.90            # 最小凸度阈值

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def load_image(path: str) -> np.ndarray:
    """加载 BGR 图像，失败返回 None"""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    """保存图像，自动创建父目录"""
    try:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        cv2.imwrite(path, img)
        return True
    except Exception:
        return False


def downscale_for_detection(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    将图像缩小至 MAX_DETECT_SIZE 以内以便快速检测。
    返回 (缩小图, 缩放因子)。若图像已足够小则返回原图和 1.0。
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_DETECT_SIZE:
        return img, 1.0
    scale = MAX_DETECT_SIZE / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale


def upscale_points(pts: np.ndarray, inv_scale: float) -> np.ndarray:
    """将检测到的角点坐标映射回原始分辨率"""
    if inv_scale == 1.0:
        return pts.astype(np.float32)
    return (pts.astype(np.float32) * inv_scale)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    将 4 个角点排序为 [左上, 右上, 右下, 左下]。
    依据：x+y 最小=左上，最大=右下；y-x 最小=右上，最大=左下。
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def compute_center_distance(pts: np.ndarray, img_w: int, img_h: int) -> float:
    """计算轮廓中心到图像中心的距离"""
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    return np.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)


def natural_sort_key(s: str):
    """自然排序键：10.jpg 排在 2.jpg 之后"""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
