"""
文档扫描仪 - 透视变换模块
将检测到的四边形角点通过透视变换矫正为矩形
"""

import numpy as np
import cv2

from .utils import WARP_PADDING


def compute_output_size(corners: np.ndarray, padding: int = WARP_PADDING) -> tuple[int, int]:
    """
    根据四个角点计算透视变换后的目标尺寸。
    宽 = max(上边长, 下边长) + 2*padding
    高 = max(左边长, 右边长) + 2*padding
    """
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bot = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width = max(int(width_top), int(width_bot)) + 2 * padding

    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    height = max(int(height_left), int(height_right)) + 2 * padding

    return width, height


def perspective_warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    对原图进行透视变换，将文档矫正为正面矩形。
    corners: 4x2 角点数组 [左上, 右上, 右下, 左下]（原始坐标）
    返回矫正后的 BGR 图像
    """
    corners = corners.astype(np.float32)
    w_out, h_out = compute_output_size(corners, WARP_PADDING)

    dst = np.array([
        [0, 0],
        [w_out - 1, 0],
        [w_out - 1, h_out - 1],
        [0, h_out - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        img, M, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped
