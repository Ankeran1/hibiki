"""
文档扫描仪 - 自动旋转矫正模块
使用投影直方图分析检测文档方向
"""

import numpy as np
import cv2


def _projection_energy(binary: np.ndarray) -> float:
    """
    计算水平投影的能量（峰值显著度）。
    文字行在水平方向产生明显的周期性峰值。
    能量 = 投影直方图的方差 / 均值，越高说明文字行方向越明确。
    """
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    # 水平投影：每行的黑色像素数
    # 假设文档是白底黑字：黑色像素 < 128
    if binary.ndim == 3:
        gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary

    proj = np.sum(gray < 128, axis=1).astype(float)

    if np.mean(proj) < 1.0:
        # 几乎没有黑色像素（空白文档）
        return 0.0

    # 能量 = 标准差 / 均值（变异系数），衡量峰值的显著程度
    mean_val = np.mean(proj)
    std_val = np.std(proj)

    # 额外检测周期性：计算自相关函数的峰值
    # 简单做法：计算投影直方图的 FFT，看主频率的能量
    proj_centered = proj - mean_val
    if np.max(np.abs(proj_centered)) < 1e-6:
        return 0.0

    fft_vals = np.abs(np.fft.rfft(proj_centered))
    # 忽略 DC 分量和最高频率
    if len(fft_vals) > 2:
        fft_vals = fft_vals[1:-1]
    peak_energy = float(np.max(fft_vals)) if len(fft_vals) > 0 else 0.0

    # 综合：变异系数 + 频域峰值能量
    energy = std_val / (mean_val + 1e-6) + peak_energy / (np.sum(proj) + 1e-6) * 10
    return energy


def detect_orientation(warped: np.ndarray) -> int:
    """
    检测矫正后图像的方向。
    对比 0°/90°/180°/270° 四个方向的投影能量，
    选择文字行方向最明显的角度。
    返回需要旋转的角度（0, 90, 180, 270），使文字正向。
    """
    h, w = warped.shape[:2]
    if h == 0 or w == 0:
        return 0

    # 如果图像太小，无法可靠判断，保持原方向
    if h < 50 or w < 50:
        return 0

    # 将图像转为灰度二值图用于分析
    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped.copy()

    # 二值化（Otsu）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 对 0°/90°/180°/270° 分别计算投影能量
    rotations = {
        0: binary,
        90: cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(binary, cv2.ROTATE_180),
        270: cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    energies = {}
    for angle, img in rotations.items():
        energies[angle] = _projection_energy(img)

    # 选择能量最高的方向
    best_angle = max(energies, key=energies.get)

    # 如果最高能量和次高能量差异不大，说明无法可靠判断
    sorted_energies = sorted(energies.values(), reverse=True)
    if len(sorted_energies) >= 2 and sorted_energies[0] < sorted_energies[1] * 1.3:
        # 方向不明确，保持原方向
        return 0

    # best_angle 是"文字行最水平"的旋转角度
    # 我们需要将图像旋转到这个角度，所以返回 best_angle
    return best_angle


def auto_rotate(warped: np.ndarray) -> tuple[np.ndarray, int]:
    """
    自动检测并矫正文档方向。
    使用投影直方图分析文字方向。
    返回 (旋转后图像, 旋转角度)
    """
    angle = detect_orientation(warped)

    if angle == 0:
        return warped, 0
    elif angle == 90:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), 90
    elif angle == 180:
        return cv2.rotate(warped, cv2.ROTATE_180), 180
    elif angle == 270:
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE), 270
    return warped, 0


def auto_rotate_binary(binary: np.ndarray, angle: int) -> np.ndarray:
    """对二值化图像应用相同的旋转"""
    if angle == 90:
        return cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(binary, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return binary
