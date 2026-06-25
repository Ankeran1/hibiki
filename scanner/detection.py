"""
文档扫描仪 - 文档边界检测模块
多策略检测 + 边缘验证 + 文档验证
"""

import math
import numpy as np
import cv2

from .utils import (
    downscale_for_detection,
    order_points,
    upscale_points,
)
from .screen_detect import detect_bright_screen
from .edge_refine import refine_corners


# ── 常量 ──────────────────────────────────────────────
MAX_DETECT_SIZE: int = 1024


def compute_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area == 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0


def _preprocess_for_edges(gray: np.ndarray) -> np.ndarray:
    """双边滤波 + CLAHE 增强"""
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)


def _sample_border_color(img_bgr: np.ndarray) -> np.ndarray:
    """
    从图像边框区域采样背景（桌面）颜色。
    取四边各 3% 宽度的条带，用中值作为背景色。
    比仅取四角更鲁棒，尤其当文档占满画面时。
    """
    h, w = img_bgr.shape[:2]
    strip = max(2, int(min(h, w) * 0.03))
    border_pixels = np.concatenate([
        img_bgr[:strip, :].reshape(-1, 3),
        img_bgr[-strip:, :].reshape(-1, 3),
        img_bgr[:, :strip].reshape(-1, 3),
        img_bgr[:, -strip:].reshape(-1, 3),
    ])
    return np.median(border_pixels, axis=0)


def _is_near_border(box: np.ndarray, sw: int, sh: int,
                    margin_ratio: float = 0.01) -> bool:
    """检查矩形是否过于贴近图像边界"""
    margin_x = max(3, int(sw * margin_ratio))
    margin_y = max(3, int(sh * margin_ratio))
    touches = sum(
        1 for pt in box
        if int(pt[0]) <= margin_x or int(pt[0]) >= sw - margin_x
        or int(pt[1]) <= margin_y or int(pt[1]) >= sh - margin_y
    )
    return touches >= 4


# ── 边缘验证：确保矩形边缘与真实图像梯度对齐 ──────────────

def _verify_edge_alignment(gray: np.ndarray, box: np.ndarray,
                            sw: int, sh: int,
                            min_alignment_ratio: float = 0.4) -> float:
    """
    验证候选矩形的四条边是否与图像中的真实梯度对齐。
    沿每条边采样，检查法线方向是否有显著梯度。
    返回对齐比例 0~1（越高越可信）。
    """
    ordered = order_points(box)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    gh, gw = gray.shape[:2]
    aligned_count = 0
    total_samples = 0

    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        edge_len = math.hypot(dx, dy)
        if edge_len < 20:
            continue

        # 法向量（指向外侧）
        nx = -(p2[1] - p1[1]) / edge_len
        ny = (p2[0] - p1[0]) / edge_len
        cx = box[:, 0].mean()
        cy = box[:, 1].mean()
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if nx * (cx - mid_x) + ny * (cy - mid_y) > 0:
            nx, ny = -nx, -ny

        # 在边两侧 ±8px 范围内寻找梯度峰值
        for t in np.linspace(0.05, 0.95, 25):
            px = p1[0] + dx * t
            py = p1[1] + dy * t

            best_grad = 0
            for offset in np.linspace(-8, 8, 9):
                ix = int(px + nx * offset)
                iy = int(py + ny * offset)
                if 0 <= ix < gw and 0 <= iy < gh:
                    g = grad_mag[iy, ix]
                    if g > best_grad:
                        best_grad = g

            total_samples += 1
            if best_grad > 20:
                aligned_count += 1

    if total_samples == 0:
        return 0.0
    return aligned_count / total_samples


def _validate_and_shrink_box(img_bgr: np.ndarray, box: np.ndarray,
                              sw: int, sh: int) -> np.ndarray:
    """
    后验证：检查框的每条边是否真的是文档边界。
    内侧采样紧贴边（+5px），外侧采样远离边（-25px），确保采到真正的背景。
    如果内外颜色差异小（< 20），说明该边不是真实文档边界，将其朝质心方向收缩。
    """
    gh, gw = img_bgr.shape[:2]
    ordered = order_points(box)
    centroid = ordered.mean(axis=0)

    new_corners = np.zeros((4, 2), dtype=np.float32)
    edge_valid = [False] * 4

    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 20:
            new_corners[i] = p1
            continue

        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        to_centroid = centroid - (p1 + p2) / 2
        if np.dot(normal, to_centroid) < 0:
            normal = -normal

        # 内侧紧贴边（+5px），外侧远离边（-25px）
        inner_colors = []
        outer_colors = []
        for t in np.linspace(0.1, 0.9, 20):
            pt = p1 + t * edge_vec
            # 内侧 5px
            inner_pt = pt + normal * 5
            ix, iy = int(round(inner_pt[0])), int(round(inner_pt[1]))
            if 0 <= ix < gw and 0 <= iy < gh:
                inner_colors.append(img_bgr[iy, ix].astype(np.float32))
            # 外侧 25px（确保采到真正的背景）
            outer_pt = pt - normal * 25
            ox, oy = int(round(outer_pt[0])), int(round(outer_pt[1]))
            if 0 <= ox < gw and 0 <= oy < gh:
                outer_colors.append(img_bgr[oy, ox].astype(np.float32))

        if len(inner_colors) < 5 or len(outer_colors) < 5:
            new_corners[i] = p1
            continue

        inner_mean = np.mean(inner_colors, axis=0)
        outer_mean = np.mean(outer_colors, axis=0)
        color_diff = float(np.linalg.norm(inner_mean - outer_mean))

        if color_diff > 20:
            edge_valid[i] = True
            new_corners[i] = p1
        else:
            edge_valid[i] = False

    # 对无效的边，朝质心方向收缩 15%
    if not all(edge_valid):
        for i in range(4):
            if not edge_valid[i]:
                direction = centroid - ordered[i]
                dist = np.linalg.norm(direction)
                if dist > 0:
                    new_corners[i] = ordered[i] + direction * 0.15
                else:
                    new_corners[i] = ordered[i]
    else:
        new_corners = ordered.copy()

    return new_corners


def _trim_box_by_color(img_bgr: np.ndarray, box: np.ndarray,
                        sw: int, sh: int,
                        max_trim_ratio: float = 0.20) -> np.ndarray:
    """
    沿每条边向内扫描颜色差异，找到真实文档边界并收缩框。
    核心：对每条边，比较"边外侧"和"边内侧"的颜色分布，
    找到颜色差异最大的位置作为真实边界。
    比梯度内缩更有效，尤其对文档与桌面颜色接近的场景。
    """
    gh, gw = img_bgr.shape[:2]
    ordered = order_points(box)
    centroid = ordered.mean(axis=0)

    new_corners = np.zeros((4, 2), dtype=np.float32)

    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 20:
            new_corners[i] = p1
            continue

        # 法向量（指向质心 = 内侧）
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        to_centroid = centroid - (p1 + p2) / 2
        if np.dot(normal, to_centroid) < 0:
            normal = -normal

        # 最大搜索距离
        max_dist = edge_len * max_trim_ratio

        # 沿边采样，计算每个偏移位置处"内侧 vs 外侧"的颜色差异
        best_offsets = []
        for t in np.linspace(0.1, 0.9, 15):
            sample_pt = p1 + t * edge_vec

            # 在法线方向上取"外侧"参考色（距离边 -15px）
            outer_pt = sample_pt - normal * 15
            ox, oy = int(round(outer_pt[0])), int(round(outer_pt[1]))
            if not (0 <= ox < gw and 0 <= oy < gh):
                continue
            outer_color = img_bgr[oy, ox].astype(np.float32)

            # 扫描内侧不同距离，找与外侧颜色差异最大的位置
            best_offset = 0
            best_diff = 0

            for d in np.linspace(2, max_dist, 20):
                inner_pt = sample_pt + d * normal
                ix, iy = int(round(inner_pt[0])), int(round(inner_pt[1]))
                if not (0 <= ix < gw and 0 <= iy < gh):
                    continue
                inner_color = img_bgr[iy, ix].astype(np.float32)
                color_diff = np.linalg.norm(inner_color - outer_color)
                if color_diff > best_diff:
                    best_diff = color_diff
                    best_offset = d

            if best_diff > 20:  # 显著颜色差异
                best_offsets.append(best_offset)

        if best_offsets:
            # 用中值偏移量，收缩到 90%（留余量）
            median_offset = float(np.median(best_offsets))
            trim = median_offset * 0.9
            new_corners[i] = p1 + normal * trim
        else:
            new_corners[i] = p1

    return new_corners


def _trim_box_by_gradient(gray: np.ndarray, box: np.ndarray,
                           sw: int, sh: int,
                           max_trim_ratio: float = 0.12) -> np.ndarray:
    """
    沿每条边向内扫描梯度，找到真实文档边界并收缩框。
    从当前边位置朝质心方向搜索，找到第一个强梯度峰值的位置。
    最大收缩比例由 max_trim_ratio 限制，避免过度收缩。
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    gh, gw = gray.shape[:2]
    ordered = order_points(box)
    centroid = ordered.mean(axis=0)

    new_corners = np.zeros((4, 2), dtype=np.float32)

    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 20:
            new_corners[i] = p1
            continue

        # 法向量（指向质心 = 内侧）
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        to_centroid = centroid - (p1 + p2) / 2
        if np.dot(normal, to_centroid) < 0:
            normal = -normal

        # 最大搜索距离
        max_dist = edge_len * max_trim_ratio

        # 沿边采样，找梯度峰值位置
        best_offsets = []
        for t in np.linspace(0.1, 0.9, 20):
            sample_pt = p1 + t * edge_vec

            best_offset = 0
            best_grad = 0

            for d in np.linspace(0, max_dist, 15):
                check_pt = sample_pt + d * normal
                ix = int(round(check_pt[0]))
                iy = int(round(check_pt[1]))
                if 0 <= ix < gw and 0 <= iy < gh:
                    g = grad_mag[iy, ix]
                    if g > best_grad:
                        best_grad = g
                        best_offset = d

            if best_grad > 15:
                best_offsets.append(best_offset)

        # 用中值偏移量收缩这条边的两个端点
        if best_offsets:
            median_offset = float(np.median(best_offsets))
            # 只收缩到偏移量的 80%，保留一些余量
            trim = median_offset * 0.8
            new_corners[i] = p1 + normal * trim
        else:
            new_corners[i] = p1

    return new_corners


# ── 评分系统 ────────────────────────────────────────────

def _document_score(gray: np.ndarray, box: np.ndarray, sw: int, sh: int,
                     edge_alignment: float) -> float:
    """
    综合评分：边缘对齐 + 内外对比度 + 面积合理性 + 长宽比。
    """
    rect = cv2.minAreaRect(box.astype(np.float32))
    area = rect[1][0] * rect[1][1]
    area_ratio = area / (sw * sh)

    # ─ 边缘对齐分（最重要）──
    # 至少 50% 的采样点有梯度 = 真实文档边缘
    align_score = edge_alignment

    # ── 内外对比度 ──
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    pts = box.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)

    interior = gray[mask > 0]
    exterior = gray[mask == 0]

    if len(interior) < 100 or len(exterior) < 100:
        return 0.0

    interior_mean = float(np.mean(interior))
    exterior_mean = float(np.mean(exterior))
    abs_diff = abs(interior_mean - exterior_mean)

    contrast_score = min(1.0, abs_diff / 40.0)

    # ── 内部均匀性（模糊文字后）──
    x, y, w, h = cv2.boundingRect(pts)
    roi = gray[max(0, y):y + h, max(0, x):x + w]
    roi_mask = mask[max(0, y):y + h, max(0, x):x + w]
    rh, rw = roi.shape[:2]
    quads_std = []
    if rh > 20 and rw > 20:
        roi_closed = cv2.morphologyEx(
            roi, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        )
        roi_mask_closed = cv2.morphologyEx(
            roi_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        )
        for qy in range(2):
            for qx in range(2):
                q = roi_closed[qy * rh // 2:(qy + 1) * rh // 2, qx * rw // 2:(qx + 1) * rw // 2]
                qm = roi_mask_closed[qy * rh // 2:(qy + 1) * rh // 2, qx * rw // 2:(qx + 1) * rw // 2]
                valid = q[qm > 0]
                if len(valid) > 50:
                    quads_std.append(float(np.std(valid)))
    interior_std = float(np.mean(quads_std)) if quads_std else 0.0

    uniformity_score = 1.0 if interior_std < 25 else (0.6 if interior_std < 45 else 0.2)

    # ─ 面积合理性：放宽上限以支持文档占满画面的情况 ──
    if area_ratio < 0.08:
        area_score = (area_ratio / 0.08) * 0.3
    elif area_ratio < 0.15:
        area_score = 0.3 + (area_ratio - 0.08) / 0.07 * 0.4
    elif area_ratio <= 0.60:
        area_score = 0.7 + (area_ratio - 0.15) / 0.45 * 0.3
    elif area_ratio <= 0.85:
        area_score = 1.0 - (area_ratio - 0.60) / 0.25 * 0.3
    elif area_ratio <= 0.95:
        area_score = 0.70 - (area_ratio - 0.85) / 0.10 * 0.30
    else:
        area_score = max(0.05, 0.40 - (area_ratio - 0.95) / 0.05 * 0.35)

    # ─ 长宽比：0.35~3.0 合理 ──
    w_r, h_r = sorted(rect[1])
    aspect = w_r / h_r if h_r > 0 else 1.0
    aspect_score = 1.0 if 0.35 <= aspect <= 3.0 else max(0.2, 1.0 - abs(math.log(aspect)) / 2)

    # 边界邻近惩罚
    border_penalty = 0.0

    # 综合评分
    return (0.25 * align_score +
            0.20 * contrast_score +
            0.15 * uniformity_score +
            0.30 * area_score +
            0.10 * aspect_score -
            border_penalty)


# ── 策略实现 ────────────────────────────────────────────

def _find_by_contours_multiscale(gray: np.ndarray, sw: int, sh: int,
                                   canny_low: int = 30, canny_high: int = 100
                                   ) -> list[tuple[np.ndarray, float]]:
    """
    Canny 多尺度轮廓检测。
    关键改进：尝试从最灵敏到保守的多组参数，找最大合理四边形。
    """
    enhanced = _preprocess_for_edges(gray)
    all_candidates = []

    # 5 组参数：极灵敏 → 灵敏 → 默认 → 保守 → 极保守
    param_sets = [
        (max(5, canny_low - 25), max(30, canny_high - 60)),
        (max(10, canny_low - 15), max(40, canny_high - 40)),
        (canny_low, canny_high),
        (min(60, canny_low + 20), min(180, canny_high + 50)),
        (min(80, canny_low + 40), min(250, canny_high + 100)),
    ]

    seen = set()

    for c_low, c_high in param_sets:
        edges = cv2.Canny(enhanced, c_low, c_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.05 * sw * sh or area > 0.90 * sw * sh:
                continue

            # 尝试多种 epsilon 找四边形
            found = False
            for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(c, eps_factor * cv2.arcLength(c, True), True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    if _is_near_border(pts, sw, sh):
                        continue
                    if compute_solidity(c) < 0.75:
                        continue

                    key = (int(pts[0, 0] / 8), int(pts[0, 1] / 8),
                           int(pts[2, 0] / 8), int(pts[2, 1] / 8))
                    if key not in seen:
                        seen.add(key)
                        all_candidates.append(pts)
                    found = True
                    break

            if not found and 5 <= len(approx) <= 8:
                hull = cv2.convexHull(c)
                box = np.array(cv2.boxPoints(cv2.minAreaRect(hull)), dtype=np.float32)
                if _is_near_border(box, sw, sh):
                    continue
                rect_area = cv2.contourArea(box.astype(np.int32).reshape(-1, 1, 2))
                ratio = area / rect_area if rect_area > 0 else 0
                if ratio > 0.75 and compute_solidity(c) > 0.80:
                    key = (int(box[0, 0] / 8), int(box[0, 1] / 8),
                           int(box[2, 0] / 8), int(box[2, 1] / 8))
                    if key not in seen:
                        seen.add(key)
                        all_candidates.append(box)

    # 评分：先算边缘对齐，再综合评分
    candidates = []
    for box in all_candidates:
        alignment = _verify_edge_alignment(gray, box, sw, sh)
        score = _document_score(gray, box, sw, sh, alignment)
        candidates.append((box, score))

    return candidates


def _find_by_adaptive_multiscale(gray: np.ndarray, sw: int, sh: int
                                   ) -> list[tuple[np.ndarray, float]]:
    """自适应阈值多尺度检测"""
    candidates = []
    seen = set()

    for block_size in [15, 21, 31]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.05 * sw * sh or area > 0.90 * sw * sh:
                continue

            for eps_factor in [0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(c, eps_factor * cv2.arcLength(c, True), True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    if _is_near_border(pts, sw, sh):
                        continue
                    if compute_solidity(c) < 0.70:
                        continue

                    key = (int(pts[0, 0] / 8), int(pts[0, 1] / 8),
                           int(pts[2, 0] / 8), int(pts[2, 1] / 8))
                    if key not in seen:
                        seen.add(key)
                        alignment = _verify_edge_alignment(gray, pts, sw, sh)
                        score = _document_score(gray, pts, sw, sh, alignment)
                        candidates.append((pts, score))
                    break

                if 5 <= len(approx) <= 8:
                    hull = cv2.convexHull(c)
                    box = np.array(cv2.boxPoints(cv2.minAreaRect(hull)), dtype=np.float32)
                    if _is_near_border(box, sw, sh):
                        continue
                    key = (int(box[0, 0] / 8), int(box[0, 1] / 8),
                           int(box[2, 0] / 8), int(box[2, 1] / 8))
                    if key not in seen:
                        seen.add(key)
                        alignment = _verify_edge_alignment(gray, box, sw, sh)
                        score = _document_score(gray, box, sw, sh, alignment)
                        candidates.append((box, score))
                    break

    return candidates


def _find_by_region_growing(img_bgr: np.ndarray, gray: np.ndarray,
                             sw: int, sh: int
                             ) -> list[tuple[np.ndarray, float]]:
    """
    基于背景排除的文档检测（不依赖边缘）。
    两阶段策略：
    1. 从图像边框采样桌面颜色（而非仅四角），适应文档占满画面的情况
    2. 用 floodfill 从四角向内填充桌面区域，反转得到文档掩码
    """
    candidates = []

    # ── 阶段1：边框颜色采样 + 颜色距离分割 ──
    bg_color = _sample_border_color(img_bgr)

    diff = img_bgr.astype(np.float32) - bg_color
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    for thresh in [25, 35, 50, 65]:
        mask_u8 = np.clip(dist * 1.2, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(mask_u8, min(int(thresh * 1.2), 255), 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        for largest in top_contours:
            area = cv2.contourArea(largest)
            if area < 0.08 * sw * sh or area > 0.92 * sw * sh:
                continue

            hull = cv2.convexHull(largest)
            box = np.array(cv2.boxPoints(cv2.minAreaRect(hull)), dtype=np.float32)

            border_ok = all(
                2 <= pt[0] <= sw - 2 and 2 <= pt[1] <= sh - 2
                for pt in box
            )
            if not border_ok:
                continue

            alignment = _verify_edge_alignment(gray, box, sw, sh)
            score = _document_score(gray, box, sw, sh, alignment)
            if score > 0.15:
                candidates.append((box, score))

    # ── 阶段2：Floodfill 从四角填充桌面，反转得文档 ──
    # 对每个角做 floodfill，标记桌面区域
    h, w = img_bgr.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    desk_mask = np.zeros((h, w), dtype=np.uint8)

    # 取四角的颜色作为 floodfill 种子
    corner_seeds = [
        (1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2),
    ]
    # 取多个种子位置的颜色的中值作为参考
    seed_colors = []
    for sy, sx in corner_seeds:
        seed_colors.append(img_bgr[sy, sx].astype(np.float32))
    # floodfill 的容差：使用较宽松的容差以覆盖光照变化
    lo_diff = (18, 18, 18)
    up_diff = (18, 18, 18)

    for sy, sx in corner_seeds:
        ff_work = ff_mask.copy()
        cv2.floodFill(img_bgr, ff_work, (sx, sy), (255, 255, 255),
                       lo_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE)
        # ff_work 中值为 1 的像素 = 被 floodfill 填充的 = 桌面
        desk_region = (ff_work[1:-1, 1:-1] == 1).astype(np.uint8) * 255
        desk_mask = cv2.bitwise_or(desk_mask, desk_region)

    # 文档 = 非桌面
    doc_mask = cv2.bitwise_not(desk_mask)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                 iterations=1)

    # 找最大连通区域
    doc_contours, _ = cv2.findContours(
        doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if doc_contours:
        # 取面积最大的 2 个
        top_doc = sorted(doc_contours, key=cv2.contourArea, reverse=True)[:2]
        for c in top_doc:
            area = cv2.contourArea(c)
            if area < 0.08 * sw * sh or area > 0.92 * sw * sh:
                continue

            hull = cv2.convexHull(c)
            box = np.array(cv2.boxPoints(cv2.minAreaRect(hull)), dtype=np.float32)

            border_ok = all(
                2 <= pt[0] <= sw - 2 and 2 <= pt[1] <= sh - 2
                for pt in box
            )
            if not border_ok:
                continue

            alignment = _verify_edge_alignment(gray, box, sw, sh)
            score = _document_score(gray, box, sw, sh, alignment)
            if score > 0.15:
                candidates.append((box, score))

    # 去重
    seen = set()
    unique = []
    for box, score in candidates:
        key = (int(box[0, 0] / 10), int(box[0, 1] / 10),
               int(box[2, 0] / 10), int(box[2, 1] / 10))
        if key not in seen:
            seen.add(key)
            unique.append((box, score))

    return unique


def _find_by_lines(gray: np.ndarray, sw: int, sh: int) -> list[tuple[np.ndarray, float]]:
    """LSD 直线检测"""
    ls = cv2.createLineSegmentDetector(0)
    lines_std = ls.detect(gray)[0]
    if lines_std is None or len(lines_std) < 4:
        return []

    lines = lines_std.reshape(-1, 4)
    lengths = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)
    min_len = max(sw, sh) * 0.12
    long_lines = lines[lengths > min_len]
    if len(long_lines) < 4:
        long_lines = lines[lengths > max(sw, sh) * 0.06]
    if len(long_lines) < 4:
        return []

    h_lines, v_lines = [], []
    for line in long_lines:
        x1, y1, x2, y2 = line
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle = 180 - angle

        # 线段两侧亮度差异检查
        dx, dy = x2 - x1, y2 - y1
        edge_len = math.hypot(dx, dy)
        nx, ny = -(dy) / edge_len, dx / edge_len
        inside_samples, outside_samples = [], []
        for t in np.linspace(0.1, 0.9, 10):
            px = x1 + dx * t
            py = y1 + dy * t
            ix, iy = int(px + nx * 8), int(py + ny * 8)
            ox, oy = int(px - nx * 8), int(py - ny * 8)
            if 0 <= iy < sh and 0 <= ix < sw:
                inside_samples.append(float(gray[iy, ix]))
            if 0 <= oy < sh and 0 <= ox < sw:
                outside_samples.append(float(gray[oy, ox]))

        line_contrast = abs(np.mean(inside_samples) - np.mean(outside_samples)) if inside_samples else 0
        if line_contrast < 10:
            continue  # 两侧亮度接近 = 同质区域内部边缘，丢弃

        x_span = abs(x2 - x1)
        y_span = abs(y2 - y1)
        is_full_span = (
            ((angle < 20 or angle > 160) and x_span > sw * 0.85) or
            (60 < angle < 120 and y_span > sh * 0.85)
        )
        if is_full_span and line_contrast < 15:
            continue

        if angle < 20 or angle > 160:
            h_lines.append(line)
        elif 60 < angle < 120:
            v_lines.append(line)

    def merge_parallel(line_list, is_h):
        if not line_list:
            return []
        positions = [(
            (l[1] + l[3]) / 2 if is_h else (l[0] + l[2]) / 2, l
        ) for l in line_list]
        positions.sort(key=lambda x: x[0])
        clusters, threshold = [], max(sw, sh) * 0.04
        for pos, line in positions:
            if not clusters or abs(pos - clusters[-1][0]) > threshold:
                clusters.append([pos, [line]])
            else:
                clusters[-1][1].append(line)
        return [cl[np.argmax([math.hypot(l[2] - l[0], l[3] - l[1]) for l in cl])]
                for _, cl in clusters]

    h_merged = merge_parallel(h_lines, True)
    v_merged = merge_parallel(v_lines, False)
    if len(h_merged) < 2 or len(v_merged) < 2:
        return []

    h_top, h_bot = h_merged[0], h_merged[-1]
    v_left, v_right = v_merged[0], v_merged[-1]

    def intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    tl, tr = intersect(h_top, v_left), intersect(h_top, v_right)
    br, bl = intersect(h_bot, v_right), intersect(h_bot, v_left)
    if any(p is None for p in [tl, tr, br, bl]):
        return []

    box = np.array([tl, tr, br, bl], dtype=np.float32)
    if _is_near_border(box, sw, sh):
        return []

    alignment = _verify_edge_alignment(gray, box, sw, sh)
    score = _document_score(gray, box, sw, sh, alignment)
    return [(box, score)]


def _find_by_outermost_lines(img_bgr: np.ndarray, gray: np.ndarray, sw: int, sh: int
                              ) -> list[tuple[np.ndarray, float]]:
    """
    用最长线段的极值位置构成外接矩形。
    专门处理文档占满画面的情况：此时桌面可见区域很小，
    常规方法失效，但文档的四条外边缘仍会产生长线段。
    关键改进：聚类线段位置，用"线段长度×簇内数量"加权，
    选权重最高的极值簇作为文档边缘，过滤桌面边缘等干扰。
    """
    ls = cv2.createLineSegmentDetector(0)
    lines_std = ls.detect(gray)[0]
    if lines_std is None or len(lines_std) < 4:
        return []

    lines = lines_std.reshape(-1, 4)
    lengths = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)

    min_len = max(sw, sh) * 0.08
    long_lines = lines[lengths > min_len]
    if len(long_lines) < 4:
        return []

    h_data, v_data = [], []
    margin = max(sw, sh) * 0.03  # 排除距图像边界太近的线段（可能是桌面边缘）
    for line in long_lines:
        x1, y1, x2, y2 = line
        # 排除线段端点距图像边界太近的（< 3%）
        if (min(x1, x2) < margin or max(x1, x2) > sw - margin or
                min(y1, y2) < margin or max(y1, y2) > sh - margin):
            continue

        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle = 180 - angle

        dx, dy = x2 - x1, y2 - y1
        edge_len = math.hypot(dx, dy)
        if edge_len < 1:
            continue
        nx, ny = -(dy) / edge_len, dx / edge_len

        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        sa, sb = [], []
        for d in [5, 10, 15]:
            for sign in [1, -1]:
                sx = int(mid_x + nx * d * sign)
                sy = int(mid_y + ny * d * sign)
                if 0 <= sx < sw and 0 <= sy < sh:
                    if sign > 0:
                        sa.append(float(gray[sy, sx]))
                    else:
                        sb.append(float(gray[sy, sx]))

        contrast = abs(np.mean(sa) - np.mean(sb)) if sa and sb else 0
        if contrast < 8:
            continue

        length = edge_len
        if angle < 25 or angle > 155:
            h_data.append(((y1 + y2) / 2, length))
        elif 65 < angle < 115:
            v_data.append(((x1 + x2) / 2, length))

    def _cluster_and_select(data, img_size):
        """聚类位置，用配对评分找最佳文档边缘"""
        if len(data) < 2:
            return None
        data.sort(key=lambda x: x[0])
        gap_threshold = img_size * 0.06

        clusters = [[data[0]]]
        for item in data[1:]:
            if item[0] - clusters[-1][-1][0] < gap_threshold:
                clusters[-1].append(item)
            else:
                clusters.append([item])

        scored = []
        for cluster in clusters:
            max_len = max(d[1] for d in cluster)
            avg_pos = sum(d[0] * d[1] for d in cluster) / sum(d[1] for d in cluster)
            scored.append((avg_pos, max_len))

        if len(scored) < 2:
            return None

        scored.sort(key=lambda x: x[0])

        # 配对评分：(左边缘长度 × 右边缘长度 × 间距) 最大的配对 = 真正的文档边缘
        # 桌面边缘通常线段较短，配对评分会自然低于真正的文档边缘配对
        best_score = 0
        best_pair = None
        n = len(scored)
        for i in range(n):
            for j in range(i + 1, n):
                pos_i, len_i = scored[i]
                pos_j, len_j = scored[j]
                dist = pos_j - pos_i
                if dist < img_size * 0.10:
                    continue
                pair_score = len_i * len_j * dist
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (pos_i, pos_j)

        if best_pair is None:
            return None

        return best_pair

    h_result = _cluster_and_select(h_data, sh)
    v_result = _cluster_and_select(v_data, sw)

    if h_result is None or v_result is None:
        return []

    y_top, y_bot = h_result
    x_left, x_right = v_result

    box = np.array([
        [x_left, y_top],
        [x_right, y_top],
        [x_right, y_bot],
        [x_left, y_bot],
    ], dtype=np.float32)

    # 裁剪到图像范围内
    for i in range(4):
        box[i][0] = max(2, min(sw - 2, box[i][0]))
        box[i][1] = max(2, min(sh - 2, box[i][1]))

    area_ratio = cv2.contourArea(box.astype(np.int32).reshape(-1, 1, 2)) / (sw * sh)
    if area_ratio < 0.10 or area_ratio > 0.95:
        return []

    # 后处理：检查框外区域是否为桌面色，如果是则收缩
    bx, by, bw_box, bh_box = cv2.boundingRect(box.astype(np.int32))
    # 取图像四角区域作为桌面参考色
    cs = max(2, int(min(sw, sh) * 0.02))
    desk_pixels = []
    for cy0, cx0 in [(0,0),(0,max(0,sw-cs)),(max(0,sh-cs),0),(max(0,sh-cs),max(0,sw-cs))]:
        region = img_bgr[cy0:cy0+cs, cx0:cx0+cs]
        if region.size > 0:
            desk_pixels.append(np.median(region.reshape(-1,3), axis=0))
    desk_color = np.median(desk_pixels, axis=0) if desk_pixels else None

    if desk_color is not None:
        # 检查每条边外侧的颜色是否接近桌面色
        ordered_tmp = order_points(box)
        for ei in range(4):
            ep1 = ordered_tmp[ei]
            ep2 = ordered_tmp[(ei+1)%4]
            e_vec = ep2 - ep1
            e_len = np.linalg.norm(e_vec)
            if e_len < 20: continue
            e_normal = np.array([e_vec[1], -e_vec[0]]) / e_len
            e_centroid = ordered_tmp.mean(axis=0)
            if np.dot(e_normal, e_centroid - (ep1+ep2)/2) > 0:
                e_normal = -e_normal

            ext_colors = []
            for t in np.linspace(0.1, 0.9, 10):
                pt = ep1 + t * e_vec
                for d in [5, 10, 20]:
                    cp = pt + e_normal * d
                    cx2, cy2 = int(round(cp[0])), int(round(cp[1]))
                    if 0 <= cx2 < sw and 0 <= cy2 < sh:
                        ext_colors.append(img_bgr[cy2, cx2].astype(np.float32))

            if len(ext_colors) >= 10:
                ext_mean = np.mean(ext_colors, axis=0)
                dist_to_desk = float(np.linalg.norm(ext_mean - desk_color))
                if dist_to_desk < 20:
                    # 外侧接近桌面色 -> 这条边太靠外了，向内收缩
                    # 找颜色变化最大的位置
                    best_d, best_diff = 0, 0
                    for d in np.linspace(10, e_len * 0.30, 15):
                        cp = ep1 + (ep2-ep1)*0.5 - e_normal * d
                        cx2, cy2 = int(round(cp[0])), int(round(cp[1]))
                        if 0 <= cx2 < sw and 0 <= cy2 < sh:
                            diff = float(np.linalg.norm(img_bgr[cy2, cx2].astype(np.float32) - ext_mean))
                            if diff > best_diff:
                                best_diff = diff
                                best_d = d
                    if best_diff > 20 and best_d > 10:
                        shift = -e_normal * best_d * 0.80
                        box[ei] = box[ei] + shift
                        ni = (ei+1)%4
                        box[ni] = box[ni] + shift

        # 重新裁剪角点
        for i in range(4):
            box[i][0] = max(2, min(sw-2, box[i][0]))
            box[i][1] = max(2, min(sh-2, box[i][1]))

    alignment = _verify_edge_alignment(gray, box, sw, sh)
    score = _document_score(gray, box, sw, sh, alignment)
    if score > 0.1:
        return [(box, score)]
    return []


def _validate_box_edges(img_bgr: np.ndarray, box: np.ndarray,
                         sw: int, sh: int) -> np.ndarray:
    """
    验证并微调框的四条边。
    保守策略：只在非常确信边位置错误时才调整，
    且调整幅度受限，避免过度收缩/扩展。
    """
    gh, gw = img_bgr.shape[:2]
    ordered = order_points(box)
    centroid = ordered.mean(axis=0)

    new_corners = ordered.copy()

    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 20:
            continue

        normal = np.array([edge_vec[1], -edge_vec[0]]) / edge_len
        to_centroid = centroid - (p1 + p2) / 2
        if np.dot(normal, to_centroid) < 0:
            normal = -normal

        # 采样多个距离
        samples = {}
        for dist in [-3, -20, 3, 15, 30]:
            samples[dist] = []

        for t in np.linspace(0.1, 0.9, 20):
            pt = p1 + t * edge_vec
            for dist in samples:
                check_pt = pt + normal * dist
                cx, cy = int(round(check_pt[0])), int(round(check_pt[1]))
                if 0 <= cx < gw and 0 <= cy < gh:
                    samples[dist].append(img_bgr[cy, cx].astype(np.float32))

        valid = {d: s for d, s in samples.items() if len(s) >= 5}
        if -3 not in valid or 3 not in valid:
            continue

        inner_3 = np.mean(valid[-3], axis=0)
        outer_3 = np.mean(valid[3], axis=0)
        cross_diff = float(np.linalg.norm(inner_3 - outer_3))

        # 外侧均匀性
        outer_uniform = True
        if 15 in valid:
            outer_uniform = float(np.linalg.norm(outer_3 - np.mean(valid[15], axis=0))) < 12
        if 30 in valid:
            outer_uniform = outer_uniform and float(np.linalg.norm(outer_3 - np.mean(valid[30], axis=0))) < 12

        # 角点贴近边界检测
        corner_min_dist = min(
            min(p1[0], gw - p1[0], p1[1], gh - p1[1]),
            min(p2[0], gw - p2[0], p2[1], gh - p2[1])
        )
        corner_critical = corner_min_dist < max(gw, gh) * 0.015

        # 判断是否跳过 - 只在非常确信时才调整
        if cross_diff > 25 and not corner_critical:
            continue

        # 角点极度贴近图像边界 -> 强制收缩
        if corner_critical:
            ref_color = outer_3 if 3 in valid else inner_3
            best_d, best_diff = 0, 0
            for d in np.linspace(15, edge_len * 0.35, 15):
                cp = p1 + (p2 - p1) * 0.5 - normal * d
                cx, cy = int(round(cp[0])), int(round(cp[1]))
                if 0 <= cx < gw and 0 <= cy < gh:
                    diff = float(np.linalg.norm(img_bgr[cy, cx].astype(np.float32) - ref_color))
                    if diff > best_diff:
                        best_diff = diff
                        best_d = d
            if best_diff > 20 and best_d > 10:
                shift = -normal * best_d * 0.80
                new_corners[i] = ordered[i] + shift
                ni = (i + 1) % 4
                new_corners[ni] = ordered[ni] + shift
            continue

        # 外侧均匀 + 跨越差异小 -> 收缩
        if outer_uniform and cross_diff < 15:
            ref_color = outer_3
            best_d, best_diff = 0, 0
            for d in np.linspace(10, edge_len * 0.30, 15):
                cp = p1 + (p2 - p1) * 0.5 - normal * d
                cx, cy = int(round(cp[0])), int(round(cp[1]))
                if 0 <= cx < gw and 0 <= cy < gh:
                    diff = float(np.linalg.norm(img_bgr[cy, cx].astype(np.float32) - ref_color))
                    if diff > best_diff:
                        best_diff = diff
                        best_d = d
            if best_diff > 25 and best_d > 10:
                max_shift = edge_len * 0.20
                actual_shift = min(best_d * 0.80, max_shift)
                shift = -normal * actual_shift
                new_corners[i] = ordered[i] + shift
                ni = (i + 1) % 4
                new_corners[ni] = ordered[ni] + shift

    for i in range(4):
        new_corners[i][0] = max(2, min(sw - 2, new_corners[i][0]))
        new_corners[i][1] = max(2, min(sh - 2, new_corners[i][1]))

    return new_corners


def detect_document(
    img: np.ndarray,
    *,
    canny_low: int = 30,
    canny_high: int = 100,
) -> tuple[np.ndarray | None, float]:
    """
    多策略文档四角检测。
    核心改进：所有候选都必须通过边缘对齐验证，
    优先选择边缘对齐好 + 面积合理的候选。
    """
    if img is None or len(img.shape) != 3:
        return None, 0.0

    small, scale = downscale_for_detection(img)
    inv_scale = 1.0 / scale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape[:2]

    # 策略0: PPT屏幕检测
    screen_corners, screen_conf = detect_bright_screen(small)
    if screen_corners is not None and screen_conf > 0.5:
        area = cv2.contourArea(screen_corners.astype(np.int32).reshape(-1, 1, 2))
        area_ratio = area / (sw * sh)
        if area_ratio < 0.55:
            mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.fillPoly(mask, [screen_corners.astype(np.int32)], 255)
            interior_mean = float(np.mean(gray[mask > 0]))
            exterior_mean = float(np.mean(gray[mask == 0])) if np.sum(mask == 0) > 0 else 0
            if interior_mean > exterior_mean + 8:
                corners = upscale_points(screen_corners, inv_scale)
                return corners, screen_conf

    # 收集所有候选
    all_candidates = []
    all_candidates.extend(_find_by_region_growing(small, gray, sw, sh))
    all_candidates.extend(_find_by_lines(gray, sw, sh))
    all_candidates.extend(_find_by_outermost_lines(small, gray, sw, sh))
    all_candidates.extend(_find_by_contours_multiscale(gray, sw, sh, canny_low, canny_high))
    all_candidates.extend(_find_by_adaptive_multiscale(gray, sw, sh))

    if not all_candidates:
        return None, 0.0

    # 处理候选：裁剪越界角点 + 面积过滤
    filtered = []
    for box, score in all_candidates:
        clipped = box.copy()
        for i in range(4):
            clipped[i][0] = max(1, min(sw - 1, clipped[i][0]))
            clipped[i][1] = max(1, min(sh - 1, clipped[i][1]))

        area_ratio = cv2.contourArea(clipped.astype(np.int32).reshape(-1, 1, 2)) / (sw * sh)
        if area_ratio < 0.10:
            continue



        alignment = _verify_edge_alignment(gray, clipped, sw, sh)
        new_score = _document_score(gray, clipped, sw, sh, alignment)
        filtered.append((clipped, new_score))

    all_candidates = filtered

    if not all_candidates:
        return None, 0.0

    # 评分选择 + 异常大候选抑制
    adjusted = []
    for box, score in all_candidates:
        area_pct = cv2.contourArea(box.astype(np.int32).reshape(-1, 1, 2)) / (sw * sh)
        bx, by, bw, bh = cv2.boundingRect(box.astype(np.int32))
        box_area_rect = bw * bh

        is_subregion = False
        for other_box, _ in all_candidates:
            if other_box is box:
                continue
            oc = other_box.copy()
            for i in range(4):
                oc[i][0] = max(1, min(sw - 1, oc[i][0]))
                oc[i][1] = max(1, min(sh - 1, oc[i][1]))
            ox, oy, ow, oh = cv2.boundingRect(oc.astype(np.int32))
            if ow * oh < box_area_rect * 2:
                continue
            inter_x = max(0, min(bx + bw, ox + ow) - max(bx, ox))
            inter_y = max(0, min(by + bh, oy + oh) - max(by, oy))
            if inter_x * inter_y / box_area_rect > 0.5:
                is_subregion = True
                break

        adjusted_score = score * 0.4 if is_subregion else score

        if area_pct > 0.70:
            adjusted_score *= 0.3
        # 大面积候选的边界检查
        if area_pct > 0.60:
            min_bp = min(min(pt[0], sw - pt[0], pt[1], sh - pt[1]) for pt in box)
            if min_bp < max(sw, sh) * 0.005:
                adjusted_score *= 0.4

        adjusted.append((box, adjusted_score, area_pct))

    by_area = sorted(adjusted, key=lambda x: -x[2])
    if len(by_area) >= 2:
        largest_area = by_area[0][2]
        second_area = by_area[1][2]
        if largest_area > second_area * 1.8 and largest_area > 0.55:
            adjusted = [
                (box, score * 0.5 if area_pct == largest_area else score, area_pct)
                for box, score, area_pct in adjusted
            ]

    adjusted.sort(key=lambda x: x[1], reverse=True)
    best_box, best_score, _ = adjusted[0]

    # 底边修正（在validate之前，用best_box小图坐标）
    corrected_box = best_box.copy()
    ordered_c = order_points(corrected_box)
    bot_y_c = max(ordered_c[1][1], ordered_c[2][1])
    bot_x1_c = min(ordered_c[1][0], ordered_c[2][0])
    bot_x2_c = max(ordered_c[1][0], ordered_c[2][0])
    bot_width_c = bot_x2_c - bot_x1_c
    if bot_width_c > 30 and bot_y_c > sh * 0.5:
        cx1c = int(max(0, bot_x1_c + bot_width_c * 0.15))
        cx2c = int(min(sw, bot_x2_c - bot_width_c * 0.15))
        # 取底边下方5px的B通道均值
        outer_y_c = min(sh - 1, int(bot_y_c) + 5)
        outer_strip_c = small[max(0,outer_y_c-1):outer_y_c+2, cx1c:cx2c]
        if outer_strip_c.size > 0:
            outer_b_c = float(np.mean(outer_strip_c[:,:,0]))
            # 从底边向上找B通道比外侧高20+的位置
            best_rise = None
            for dy in range(-3, -int(sh*0.4), -2):
                check_y = int(bot_y_c) + dy
                if check_y < 0: break
                strip = small[max(0,check_y-1):check_y+2, cx1c:cx2c]
                if strip.size > 0:
                    cb = float(np.mean(strip[:,:,0]))
                    if cb > outer_b_c + 20:
                        best_rise = check_y
                        break
            if best_rise is not None:
                # 找过渡点
                trans = best_rise
                for dy in range(0, int(bot_y_c - best_rise) + 5):
                    cy2 = best_rise + dy
                    if cy2 >= sh: break
                    strip = small[max(0,cy2-1):cy2+2, cx1c:cx2c]
                    if strip.size > 0:
                        cb = float(np.mean(strip[:,:,0]))
                        if cb < outer_b_c + 10:
                            trans = max(best_rise, cy2 - 3)
                            break
                if trans < bot_y_c - 3:
                    for ci in [1, 2]:
                        corrected_box[ci][1] = min(corrected_box[ci][1], float(trans))

    # 边缘验证微调
    validated = _validate_box_edges(small, corrected_box, sw, sh)

    # 边缘精修
    refined = refine_corners(gray, order_points(validated))
    corners = upscale_points(refined, inv_scale)
    return corners, min(1.0, best_score)



def detect_and_draw(img: np.ndarray, canny_low: int = 30, canny_high: int = 100) -> tuple[np.ndarray, np.ndarray | None]:
    """检测文档并在原图上绘制角点（调试用）"""
    annotated = img.copy()
    corners, conf = detect_document(img, canny_low=canny_low, canny_high=canny_high)
    if corners is not None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 4)
        for i, (x, y) in enumerate(corners):
            cv2.circle(annotated, (int(x), int(y)), 10, (0, 0, 255), -1)
            labels = ['左上', '右上', '右下', '左下']
            cv2.putText(annotated, labels[i], (int(x) + 12, int(y) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return annotated, corners
