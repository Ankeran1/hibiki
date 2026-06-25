"""
文档扫描仪 - GUI 图像画布组件
支持缩放、平移、角点叠加显示
"""

import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2


class ImageCanvas(tk.Canvas):
    """可缩放平移的图像画布，支持角点叠加显示"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#e0e0e0", highlightthickness=1,
                         highlightbackground="#bdbdbd", **kwargs)
        self._original_img = None      # 原始 BGR 图像 (numpy)
        self._display_img = None       # 当前显示的 PIL Image
        self._photo = None             # PhotoImage 引用
        self._corners = None           # 4x2 角点 (原始坐标)
        self._view_mode = "original"   # "original" / "warped" / "binary"

        # 缩放平移状态
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None

        # 绑定事件
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Configure>", self._on_resize)

    def set_image(self, img_bgr: np.ndarray | None, corners: np.ndarray | None = None):
        """设置图像并显示"""
        self._original_img = img_bgr
        self._corners = corners
        if img_bgr is None:
            self.delete("all")
            self._display_img = None
            self._photo = None
            return
        self._apply_view()

    def set_view_mode(self, mode: str):
        """切换视图模式: original / warped / binary"""
        self._view_mode = mode
        self._apply_view()

    def _apply_view(self):
        """根据当前视图模式渲染图像"""
        if self._original_img is None:
            return

        img = self._original_img

        # 如果有角点且是原图模式，绘制角点叠加
        if self._view_mode == "original" and self._corners is not None:
            img = self._draw_corners(img)

        # 转为 RGB 然后 PIL
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 缩放到画布大小
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        # 计算适配缩放
        fit_scale = min(canvas_w / pil_img.width, canvas_h / pil_img.height)
        self._scale = fit_scale * self._zoom_level if hasattr(self, '_zoom_level') else fit_scale
        self._zoom_level = getattr(self, '_zoom_level', 1.0)

        new_w = max(1, int(pil_img.width * self._scale))
        new_h = max(1, int(pil_img.height * self._scale))
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self._display_img = resized
        self._photo = ImageTk.PhotoImage(resized)

        self.delete("image", "corner_overlay")
        img_id = self.create_image(0, 0, anchor="nw", image=self._photo, tags="image")

        # 居中
        self._offset_x = (canvas_w - new_w) // 2
        self._offset_y = (canvas_h - new_h) // 2
        self.coords("image", self._offset_x, self._offset_y)

    def _draw_corners(self, img: np.ndarray) -> np.ndarray:
        """在图像上绘制角点和边框"""
        annotated = img.copy()
        if self._corners is not None:
            pts = self._corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 4)
            for i, (x, y) in enumerate(self._corners):
                cv2.circle(annotated, (int(x), int(y)), 10, (0, 0, 255), -1)
        return annotated

    def _on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        if self._original_img is None:
            return
        delta = 1.1 if event.delta > 0 else 0.9
        self._zoom_level = getattr(self, '_zoom_level', 1.0) * delta
        self._zoom_level = max(0.3, min(4.0, self._zoom_level))
        self._apply_view()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None or self._display_img is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._offset_x += dx
        self._offset_y += dy
        self._drag_start = (event.x, event.y)
        self.move("image", dx, dy)

    def _on_resize(self, event):
        if self._original_img is not None:
            self._apply_view()

    def get_display_scale(self) -> float:
        """获取当前显示缩放比例"""
        return self._scale


class CornerAdjustCanvas(tk.Canvas):
    """可拖拽调节四角的画布，用于手动校正文档边界"""

    CORNER_RADIUS = 8
    CORNER_COLORS = ["#FF4444", "#FF8800", "#44AA44", "#4488FF"]
    CORNER_LABELS = ["左上", "右上", "右下", "左下"]

    def __init__(self, parent, img_bgr: np.ndarray, corners: np.ndarray,
                 on_corner_changed=None, **kwargs):
        super().__init__(parent, bg="#333333", highlightthickness=0, **kwargs)
        self._img = img_bgr.copy()
        self._corners = corners.copy().astype(np.float32)
        self._on_change = on_corner_changed
        self._dragging = -1
        self._display_scale = 1.0
        self._zoom_level = 1.0
        self._center_offset = (0, 0)
        self._user_pan = (0, 0)
        self._panning = False
        self._pan_start = (0, 0)

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_zoom)
        self.bind("<Configure>", lambda e: self._draw())

        self._draw()

    def _draw(self):
        self.delete("all")
        h, w = self._img.shape[:2]
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 10 or ch < 10:
            return

        # 计算适配缩放（考虑缩放级别）
        base_scale = min(cw / w, ch / h) * 0.95
        self._display_scale = base_scale * self._zoom_level
        dw = max(1, int(w * self._display_scale))
        dh = max(1, int(h * self._display_scale))
        self._center_offset = ((cw - dw) // 2, (ch - dh) // 2)

        # 总偏移 = 居中偏移 + 用户平移偏移
        ox = self._center_offset[0] + self._user_pan[0]
        oy = self._center_offset[1] + self._user_pan[1]

        # 绘制图像
        rgb = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.create_image(ox, oy, anchor="nw", image=self._photo, tags="img")

        # 绘制绿色四边形
        cpts = [self._to_canvas(c) for c in self._corners]
        for i in range(4):
            j = (i + 1) % 4
            self.create_line(cpts[i][0], cpts[i][1], cpts[j][0], cpts[j][1],
                             fill="#00FF00", width=2, tags="quad")

        # 绘制可拖拽角点（小圆点，精细调整）
        r = self.CORNER_RADIUS
        for i, (cx, cy) in enumerate(cpts):
            self.create_oval(cx - r, cy - r, cx + r, cy + r,
                             fill=self.CORNER_COLORS[i],
                             outline="white", width=1, tags=f"c{i}")

    def _on_zoom(self, event):
        """滚轮缩放"""
        delta = 1.15 if event.delta > 0 else 0.87
        self._zoom_level *= delta
        self._zoom_level = max(0.5, min(5.0, self._zoom_level))
        self._draw()

    def _to_canvas(self, pt):
        ox = self._center_offset[0] + self._user_pan[0]
        oy = self._center_offset[1] + self._user_pan[1]
        return (pt[0] * self._display_scale + ox,
                pt[1] * self._display_scale + oy)

    def _to_image(self, cx, cy):
        ox = self._center_offset[0] + self._user_pan[0]
        oy = self._center_offset[1] + self._user_pan[1]
        return np.array([
            (cx - ox) / self._display_scale,
            (cy - oy) / self._display_scale
        ])

    def _on_press(self, event):
        for i, c in enumerate(self._corners):
            px, py = self._to_canvas(c)
            if math.hypot(event.x - px, event.y - py) <= self.CORNER_RADIUS + 5:
                self._dragging = i
                return
        self._dragging = -1
        # 未命中角点 → 进入平移模式
        self._panning = True
        self._pan_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._panning:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self._user_pan = (self._user_pan[0] + dx,
                              self._user_pan[1] + dy)
            self._pan_start = (event.x, event.y)
            self.move("all", dx, dy)
            return
        if self._dragging < 0:
            return
        new_pt = self._to_image(event.x, event.y)
        h, w = self._img.shape[:2]
        new_pt[0] = max(0, min(w - 1, new_pt[0]))
        new_pt[1] = max(0, min(h - 1, new_pt[1]))
        self._corners[self._dragging] = new_pt
        self._draw()
        if self._on_change:
            self._on_change(self._corners.copy())

    def _on_release(self, event):
        self._dragging = -1
        self._panning = False

    def get_corners(self) -> np.ndarray:
        return self._corners.copy()

    def reset_corners(self, corners: np.ndarray):
        self._corners = corners.copy().astype(np.float32)
        self._draw()
