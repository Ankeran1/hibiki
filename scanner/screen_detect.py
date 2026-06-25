"""
文档扫描仪 - PPT屏幕/亮屏检测模块
检测暗背景中的明亮矩形区域（如电脑屏幕上的PPT）
"""

import numpy as np
import cv2

from .utils import order_points


def detect_bright_screen(
    img_bgr: np.ndarray,
    *,
    brightness_threshold: int = 140,
    min_contrast: int = 40,
) -> tuple[np.ndarray | None, float]:
    """
    检测暗背景中的明亮屏幕区域。
    返回 (4x2 角点数组, 置信度) 或 (None, 0.0)
    """
    if img_bgr is None or len(img_bgr.shape) != 3:
        return None, 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # ── 步骤1：检查图像边缘是否为暗背景 ──
    border = int(min(h, w) * 0.04)
    border_pixels = np.concatenate([
        gray[:border, :].flatten(),
        gray[-border:, :].flatten(),
        gray[:, :border].flatten(),
        gray[:, -border:].flatten(),
    ])
    border_mean = float(np.mean(border_pixels))

    # 如果边缘不够暗，不是屏幕场景
    if border_mean > 130:
        return None, 0.0

    # ── 步骤2：亮度阈值分割 ──
    _, bright_mask = cv2.threshold(
        gray, brightness_threshold, 255, cv2.THRESH_BINARY
    )

    # ─ 步骤3：形态学闭运算连接亮区 ──
    # 使用较小核避免过度扩张到墙壁等背景
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    # 开运算去除小噪点
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    # ── 步骤4：查找最大轮廓 ──
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 0.05 * h * w:
        return None, 0.0

    # ── 步骤5：四边形近似或外接矩形 ──
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        corners = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)

    # ── 步骤6：验证内部亮、外部暗 ──
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    interior_mean = float(np.mean(gray[mask > 0]))

    exterior_mask = cv2.bitwise_not(mask)
    exterior_pixels = gray[exterior_mask > 0]
    exterior_mean = float(np.mean(exterior_pixels)) if len(exterior_pixels) > 0 else 0

    contrast = interior_mean - exterior_mean
    if contrast < min_contrast:
        return None, 0.0

    ordered = order_points(corners)
    confidence = min(1.0, contrast / 120.0)

    return ordered, confidence
