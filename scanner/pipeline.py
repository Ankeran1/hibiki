"""
文档扫描仪 - 扫描流水线模块
整合检测、透视变换、旋转矫正、二值化为一站式接口
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from .utils import load_image, save_image
from .detection import detect_document
from .transform import perspective_warp
from .binarize import enhance_scanner_output


@dataclass
class ScanResult:
    """扫描结果"""
    success: bool = False
    warped: np.ndarray | None = None          # 透视矫正彩色图
    binarized: np.ndarray | None = None       # 二值化输出
    corners: np.ndarray | None = None         # 原始坐标下的四角点
    confidence: float = 0.0                   # 检测置信度 0~1
    error_message: str = ""                   # 错误信息（中文）


def scan_image(
    img: np.ndarray,
    *,
    binarize_method: str = "auto",
    canny_low: int = 30,
    canny_high: int = 100,
    manual_corners: np.ndarray | None = None,
) -> ScanResult:
    """
    完整扫描流水线。
    manual_corners: 手动指定的角点 (4x2)，跳过自动检测
    """
    if img is None or len(img.shape) != 3:
        return ScanResult(success=False, error_message="无法读取图像文件")

    # 检测文档四角（或使用手动角点）
    if manual_corners is not None:
        corners = manual_corners.astype(np.float32)
        confidence = 1.0
    else:
        corners, confidence = detect_document(
            img, canny_low=canny_low, canny_high=canny_high
        )

    if corners is None:
        return ScanResult(
            success=False,
            error_message="未检测到文档边界，请尝试调整Canny阈值或手动调节角点"
        )

    # 透视矫正
    try:
        warped = perspective_warp(img, corners)
    except Exception as e:
        return ScanResult(
            success=False, corners=corners, confidence=confidence,
            error_message=f"透视变换失败: {str(e)}",
        )

    # 二值化
    try:
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binarized = enhance_scanner_output(warped_gray, method=binarize_method)
    except Exception as e:
        return ScanResult(
            success=False, warped=warped, corners=corners, confidence=confidence,
            error_message=f"二值化失败: {str(e)}",
        )

    return ScanResult(
        success=True,
        warped=warped,
        binarized=binarized,
        corners=corners,
        confidence=confidence,
    )


def scan_file(
    input_path: str,
    output_path: str,
    *,
    binarize_method: str = "auto",
    canny_low: int = 30,
    canny_high: int = 100,
    save_warped: bool = True,
    save_binarized: bool = True,
    manual_corners: np.ndarray | None = None,
) -> ScanResult:
    img = load_image(input_path)
    if img is None:
        return ScanResult(success=False, error_message="无法读取图像文件")

    result = scan_image(
        img,
        binarize_method=binarize_method,
        canny_low=canny_low,
        canny_high=canny_high,
        manual_corners=manual_corners,
    )

    if not result.success:
        return result

    if save_warped and result.warped is not None:
        stem = Path(input_path).stem
        warped_path = str(Path(output_path).parent / f"{stem}_warped.jpg")
        save_image(warped_path, result.warped)

    if save_binarized and result.binarized is not None:
        save_image(output_path, result.binarized)

    return result


def scan_file_to_dir(
    input_path: str,
    output_dir: str,
    *,
    binarize_method: str = "auto",
    canny_low: int = 30,
    canny_high: int = 100,
) -> ScanResult:
    stem = Path(input_path).stem
    binary_out = os.path.join(output_dir, f"{stem}_binarized.png")
    return scan_file(
        input_path, binary_out,
        binarize_method=binarize_method,
        canny_low=canny_low,
        canny_high=canny_high,
        save_warped=True,
        save_binarized=True,
    )
