"""
文档扫描仪 - 边缘精修模块
通过梯度搜索将候选矩形的边对齐到真实文档边界
关键：搜索偏向内侧，防止桌面梯度把角点推出
支持自适应搜索半径和子像素级精修
"""

import numpy as np
import cv2


def _line_intersection(
    line1: tuple[np.ndarray, np.ndarray],
    line2: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)


def _subpixel_refine(grad_mag: np.ndarray, best_pos: np.ndarray,
                     normal: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """
    在梯度最大点附近进行抛物线插值，获得亚像素精度的位置。
    在法线方向上取 best_pos ± 1 的三个采样点，拟合抛物线求极值。
    """
    best_g = 0.0
    best_p = best_pos.copy()

    # 在法线方向上取 -1, 0, +1 三个亚像素位置
    for offset in [-0.5, 0.0, 0.5]:
        check_pt = best_pos + offset * normal
        ix = int(round(check_pt[0]))
        iy = int(round(check_pt[1]))
        if 0 <= ix < gw and 0 <= iy < gh:
            g = grad_mag[iy, ix]
            if g > best_g:
                best_g = g
                best_p = check_pt.copy()

    return best_p


def refine_corners(
    gray: np.ndarray,
    corners: np.ndarray,
    *,
    search_radius: int | None = None,
    sample_count: int = 35,
    min_gradient: float = 15,
) -> np.ndarray:
    """
    精修候选矩形的四条边。
    搜索偏向内侧（朝向质心），外侧搜索范围仅为内侧的30%。
    搜索半径根据图像分辨率自适应。
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    gh, gw = gray.shape[:2]
    centroid = corners.mean(axis=0)

    # 自适应搜索半径：根据图像分辨率
    if search_radius is None:
        diag = np.sqrt(gw ** 2 + gh ** 2)
        if diag > 1500:
            search_radius = 25
        elif diag > 800:
            search_radius = 18
        else:
            search_radius = 12

    edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
    new_lines = []

    for i_start, i_end in edge_indices:
        p1 = corners[i_start]
        p2 = corners[i_end]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)

        if edge_len < 20:
            new_lines.append((p1.copy(), p2.copy()))
            continue

        # 法向量：确保指向质心（内侧）
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        to_centroid = centroid - ((p1 + p2) / 2)
        if np.dot(normal, to_centroid) < 0:
            normal = -normal

        # 搜索范围：内侧 search_radius，外侧 30%
        out_range = int(search_radius * 0.3)
        offsets = np.linspace(-out_range, search_radius, out_range + search_radius + 1)

        gradient_points = []

        for t in np.linspace(0.05, 0.95, sample_count):
            sample_pt = p1 + t * edge_vec
            best_pos = None
            best_grad = 0

            for offset in offsets:
                check_pt = sample_pt + offset * normal
                ix = int(round(check_pt[0]))
                iy = int(round(check_pt[1]))

                if 0 <= ix < gw and 0 <= iy < gh:
                    g = grad_mag[iy, ix]
                    if g > best_grad:
                        best_grad = g
                        best_pos = check_pt.copy()

            if best_pos is not None and best_grad > min_gradient:
                # 子像素精修
                refined_pos = _subpixel_refine(grad_mag, best_pos, normal, gw, gh)
                gradient_points.append(refined_pos)

        # fitLine 拟合
        if len(gradient_points) >= max(5, sample_count // 3):
            pts_arr = np.array(gradient_points, dtype=np.float32)
            params = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            vx, vy, cx, cy = params
            extent = edge_len * 0.8
            lp = np.array([cx - vx * extent, cy - vy * extent], dtype=np.float32)
            rp = np.array([cx + vx * extent, cy + vy * extent], dtype=np.float32)
            new_lines.append((lp, rp))
        else:
            new_lines.append((p1.copy(), p2.copy()))

    # 相邻边求交
    refined = np.zeros((4, 2), dtype=np.float32)
    border = max(5, int(min(gw, gh) * 0.015))
    for i in range(4):
        result = _line_intersection(new_lines[i], new_lines[(i + 1) % 4])
        if result is not None:
            rx = float(max(border, min(gw - 1 - border, result[0])))
            ry = float(max(border, min(gh - 1 - border, result[1])))
            refined[i] = [rx, ry]
        else:
            refined[i] = corners[i]

    return refined
