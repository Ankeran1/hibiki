"""
文档扫描仪 - 主界面
基于 tkinter 的文档扫描 GUI 应用
"""

import os
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from scanner.pipeline import scan_image, scan_file_to_dir, ScanResult
from scanner.detection import detect_document
from scanner.transform import perspective_warp
from scanner.binarize import enhance_scanner_output
from scanner.batch import batch_process
from .widgets import ImageCanvas, CornerAdjustCanvas
from .dialogs import BatchProgressDialog, BatchFileSelectDialog


class App:
    """文档扫描仪主应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("文档扫描仪 v1.0")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.configure(bg="#f5f5f5")

        self._current_img = None
        self._warped_img = None
        self._binary_img = None
        self._corners = None
        self._current_path = ""
        self._confidence = 0.0

        self._build_ui()

    # ── UI 构建 ──────────────────────────────────────────

    def _build_ui(self):
        # 标题栏
        title_frame = tk.Frame(self.root, bg="#1976D2", height=50)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text=" 文档扫描仪",
                 font=("微软雅黑", 16, "bold"), fg="white", bg="#1976D2",
                 anchor="w").pack(side="left", padx=15, pady=8)
        tk.Label(title_frame, text="Document Scanner",
                 font=("微软雅黑", 9), fg="#bbdefb", bg="#1976D2",
                 anchor="w").pack(side="left", pady=12)

        # 主内容区
        main_paned = tk.PanedWindow(self.root, orient="horizontal",
                                     bg="#f5f5f5")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # 左侧：图像预览
        left_frame = tk.Frame(main_paned, bg="#f5f5f5")
        main_paned.add(left_frame, minsize=400)

        self.canvas = ImageCanvas(left_frame)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # 视图切换
        view_frame = tk.Frame(left_frame, bg="#f5f5f5")
        view_frame.pack(fill="x", padx=5, pady=(0, 5))

        self._view_var = tk.StringVar(value="original")
        tk.Radiobutton(view_frame, text="原图", variable=self._view_var,
                        value="original", command=self._on_view_change,
                        font=("微软雅黑", 9), bg="#f5f5f5").pack(side="left", padx=10)
        tk.Radiobutton(view_frame, text="矫正图", variable=self._view_var,
                        value="warped", command=self._on_view_change,
                        font=("微软雅黑", 9), bg="#f5f5f5").pack(side="left", padx=10)
        tk.Radiobutton(view_frame, text="二值化", variable=self._view_var,
                        value="binary", command=self._on_view_change,
                        font=("微软雅黑", 9), bg="#f5f5f5").pack(side="left", padx=10)

        # 右侧：控制面板
        right_frame = tk.Frame(main_paned, bg="#f5f5f5", width=320)
        main_paned.add(right_frame, minsize=280)

        self._make_group(right_frame, "文件操作", self._build_file_group)
        self._make_group(right_frame, "二值化方法", self._build_method_group)
        self._make_group(right_frame, "处理与输出", self._build_action_group)
        self._make_group(right_frame, "参数调节", self._build_param_group)

        # 状态栏
        status_frame = tk.Frame(self.root, bg="#e0e0e0", height=28)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        self._lbl_status = tk.Label(status_frame, text="  就绪 | 等待操作",
                                     font=("微软雅黑", 9), bg="#e0e0e0",
                                     anchor="w")
        self._lbl_status.pack(fill="both")

    def _make_group(self, parent, title, builder):
        group = tk.LabelFrame(parent, text=title,
                               font=("微软雅黑", 9, "bold"),
                               bg="#f5f5f5", padx=8, pady=6)
        group.pack(fill="x", padx=8, pady=4)
        builder(group)

    def _build_file_group(self, parent):
        btn_frame = tk.Frame(parent, bg="#f5f5f5")
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text=" 打开图片", command=self._on_open_file,
                   width=14, font=("微软雅黑", 9)).pack(side="left", padx=2, pady=2)
        tk.Button(btn_frame, text=" 批量处理", command=self._on_batch,
                   width=14, font=("微软雅黑", 9)).pack(side="left", padx=2, pady=2)

    def _build_method_group(self, parent):
        self._method_var = tk.StringVar(value="auto")
        for val, text in [
            ("auto", "自动选择（推荐）"),
            ("otsu", "Otsu 法"),
            ("adaptive", "自适应阈值"),
            ("sauvola", "Sauvola 法"),
        ]:
            tk.Radiobutton(parent, text=text, variable=self._method_var,
                            value=val, font=("微软雅黑", 9),
                            bg="#f5f5f5").pack(anchor="w", pady=1)

    def _build_action_group(self, parent):
        self._btn_scan = tk.Button(parent, text=" 开始扫描",
                                    command=self._on_scan,
                                    bg="#4CAF50", fg="white",
                                    font=("微软雅黑", 12, "bold"),
                                    height=2, width=20)
        self._btn_scan.pack(pady=5)
        tk.Button(parent, text=" 保存结果", command=self._on_save,
                   width=20, font=("微软雅黑", 9)).pack(pady=3)

        # 旋转纠正按钮
        rotate_frame = tk.Frame(parent, bg="#f5f5f5")
        rotate_frame.pack(fill="x", pady=3)
        tk.Button(rotate_frame, text="↩ 逆时针旋转90°",
                   command=self._on_rotate_ccw,
                   width=14, font=("微软雅黑", 9)).pack(side="left", padx=2)
        tk.Button(rotate_frame, text="↪ 顺时针旋转90°",
                   command=self._on_rotate_cw,
                   width=14, font=("微软雅黑", 9)).pack(side="left", padx=2)

        self._lbl_output = tk.Label(parent, text="", font=("微软雅黑", 8),
                                     fg="gray", wraplength=260, justify="left")
        self._lbl_output.pack(pady=2)

    def _build_param_group(self, parent):
        f1 = tk.Frame(parent, bg="#f5f5f5")
        f1.pack(fill="x", pady=2)
        tk.Label(f1, text="Canny 低阈值:", font=("微软雅黑", 9),
                  bg="#f5f5f5").pack(side="left")
        self._scale_low = tk.Scale(f1, from_=10, to=100, orient="horizontal",
                                    resolution=5, font=("微软雅黑", 8),
                                    bg="#f5f5f5", length=160)
        self._scale_low.set(30)
        self._scale_low.pack(side="right")

        f2 = tk.Frame(parent, bg="#f5f5f5")
        f2.pack(fill="x", pady=2)
        tk.Label(f2, text="Canny 高阈值:", font=("微软雅黑", 9),
                  bg="#f5f5f5").pack(side="left")
        self._scale_high = tk.Scale(f2, from_=50, to=300, orient="horizontal",
                                     resolution=10, font=("微软雅黑", 8),
                                     bg="#f5f5f5", length=160)
        self._scale_high.set(100)
        self._scale_high.pack(side="right")

    # ── 事件处理 ──────────────────────────────────────────

    def _on_open_file(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                        ("所有文件", "*.*")]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "无法读取图像文件")
            return
        self._current_img = img
        self._warped_img = None
        self._binary_img = None
        self._corners = None
        self._current_path = path
        self._view_var.set("original")
        self.canvas.set_image(img)
        self._set_status(f"已加载: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    def _on_scan(self):
        if self._current_img is None:
            messagebox.showinfo("提示", "请先打开一张图片")
            return

        self._set_status("正在检测文档边界...")
        self.root.update_idletasks()

        canny_low = self._scale_low.get()
        canny_high = self._scale_high.get()

        # 自动检测角点
        corners, confidence = detect_document(
            self._current_img, canny_low=canny_low, canny_high=canny_high
        )

        if corners is None:
            self._set_status("检测失败: 未找到文档边界")
            messagebox.showwarning("检测失败",
                "未检测到文档边界。\n请尝试调整Canny阈值后重试。")
            return

        self._corners = corners
        self._confidence = confidence

        # 显示角点确认对话框
        self._show_corner_dialog(corners, confidence, canny_low, canny_high)

    def _show_corner_dialog(self, corners, confidence, canny_low, canny_high):
        """显示角点调节对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("确认文档角点")
        dialog.geometry("1050x680")
        dialog.transient(self.root)
        dialog.configure(bg="#2c2c2c")

        # 左侧：可拖拽画布
        left = tk.Frame(dialog, bg="#2c2c2c")
        left.pack(side="left", fill="both", expand=True)

        adjust_canvas = CornerAdjustCanvas(
            left, self._current_img, corners,
            on_corner_changed=lambda c: self._update_preview(adjust_canvas, preview_lbl)
        )
        adjust_canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # 右侧：预览 + 按钮
        right = tk.Frame(dialog, bg="#2c2c2c", width=290)
        right.pack(side="right", fill="y", padx=5, pady=5)
        right.pack_propagate(False)

        tk.Label(right, text="实时预览", font=("微软雅黑", 10, "bold"),
                 fg="white", bg="#2c2c2c").pack(pady=5)

        preview_lbl = tk.Label(right, bg="#444", width=260, height=360)
        preview_lbl.pack(pady=5)

        tk.Label(right,
                 text=f"置信度: {confidence:.0%}\n拖拽红色角点调整边界",
                 font=("微软雅黑", 8), fg="#aaa", bg="#2c2c2c",
                 justify="center").pack(pady=5)

        # 按钮
        btn_frame = tk.Frame(right, bg="#2c2c2c")
        btn_frame.pack(fill="x", pady=10)

        tk.Button(btn_frame, text="✅ 确认角点",
                   command=lambda: self._finalize_corners(
                       adjust_canvas.get_corners(), dialog, canny_low, canny_high),
                   bg="#4CAF50", fg="white",
                   font=("微软雅黑", 10, "bold"), width=16
                   ).pack(fill="x", pady=3)

        tk.Button(btn_frame, text="🔄 重置为自动检测",
                   command=lambda: adjust_canvas.reset_corners(corners),
                   font=("微软雅黑", 9), width=16
                   ).pack(fill="x", pady=3)

        tk.Button(btn_frame, text="❌ 取消",
                   command=dialog.destroy,
                   font=("微软雅黑", 9), width=16
                   ).pack(fill="x", pady=3)

        # 初始预览
        self._update_preview(adjust_canvas, preview_lbl)

        # 居中
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _update_preview(self, adjust_canvas, preview_lbl):
        """更新预览图"""
        corners = adjust_canvas.get_corners()
        try:
            warped = perspective_warp(self._current_img, corners)
            # 缩放到预览大小
            small = cv2.resize(warped, (240, 320))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            preview_lbl.config(image=photo)
            preview_lbl._photo = photo
        except Exception:
            pass

    def _finalize_corners(self, corners, dialog, canny_low, canny_high):
        """确认角点后执行完整扫描"""
        dialog.destroy()
        self._set_status("正在矫正和二值化...")
        self.root.update_idletasks()

        method = self._method_var.get()

        # 透视矫正
        warped = perspective_warp(self._current_img, corners)

        # 二值化
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binarized = enhance_scanner_output(warped_gray, method=method)

        self._warped_img = warped
        self._binary_img = binarized
        self._corners = corners

        # 在原图上标注角点
        annotated = self._current_img.copy()
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 4)
        for i, (x, y) in enumerate(corners):
            cv2.circle(annotated, (int(x), int(y)), 8, (0, 0, 255), -1)
        self._annotated_img = annotated

        self._show_current_view()

        self._set_status(
            f"扫描完成 | 置信度: {self._confidence:.0%} | "
            f"矫正尺寸: {warped.shape[1]}x{warped.shape[0]}"
        )

    def _show_current_view(self):
        mode = self._view_var.get()
        if mode == "original":
            self.canvas.set_image(
                getattr(self, '_annotated_img', self._current_img),
                self._corners
            )
        elif mode == "warped" and self._warped_img is not None:
            self.canvas.set_image(self._warped_img)
        elif mode == "binary" and self._binary_img is not None:
            binary_bgr = cv2.cvtColor(self._binary_img, cv2.COLOR_GRAY2BGR)
            self.canvas.set_image(binary_bgr)

    def _on_view_change(self):
        self._show_current_view()

    def _rotate_current_result(self, angle_cv2):
        """旋转当前处理结果（矫正图 + 二值化图）"""
        if self._warped_img is None and self._binary_img is None:
            messagebox.showinfo("提示", "请先扫描后再旋转")
            return
        if self._warped_img is not None:
            self._warped_img = cv2.rotate(self._warped_img, angle_cv2)
        if self._binary_img is not None:
            self._binary_img = cv2.rotate(self._binary_img, angle_cv2)
        self._show_current_view()

    def _on_rotate_cw(self):
        self._rotate_current_result(cv2.ROTATE_90_CLOCKWISE)

    def _on_rotate_ccw(self):
        self._rotate_current_result(cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _on_save(self):
        mode = self._view_var.get()
        if mode == "warped" and self._warped_img is not None:
            img_to_save = self._warped_img
            ext = ".jpg"
        elif mode == "binary" and self._binary_img is not None:
            img_to_save = self._binary_img
            ext = ".png"
        else:
            messagebox.showinfo("提示", "请先扫描并切换到矫正图或二值化视图")
            return

        default_name = Path(self._current_path).stem + ("_warped" if mode == "warped" else "_binarized")
        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=ext,
            initialfile=default_name + ext,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")] if ext == ".jpg" else [("PNG", "*.png")]
        )
        if path:
            cv2.imwrite(path, img_to_save)
            self._lbl_output.config(text=f"已保存: {path}")
            self._set_status(f"已保存: {os.path.basename(path)}")

    def _on_batch(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return

        # 显示文件选择对话框
        select_dialog = BatchFileSelectDialog(self.root, folder)
        self.root.wait_window(select_dialog)
        selected_files = select_dialog.get_selected_files()
        if not selected_files:
            return

        output_dir = filedialog.askdirectory(title="选择输出文件夹")
        if not output_dir:
            return

        dialog = BatchProgressDialog(self.root, len(selected_files))
        dialog.update_idletasks()

        method = self._method_var.get()
        canny_low = self._scale_low.get()
        canny_high = self._scale_high.get()

        def worker():
            from scanner.pipeline import scan_file_to_dir

            def on_progress(current, total, filename, success):
                self.root.after(0, dialog.update_progress, current, total, filename, success)

            succeeded, failed = 0, 0
            for idx, filename in enumerate(selected_files, 1):
                input_path = os.path.join(folder, filename)
                result = scan_file_to_dir(
                    input_path, output_dir,
                    binarize_method=method,
                    canny_low=canny_low,
                    canny_high=canny_high,
                )
                if result.success:
                    succeeded += 1
                else:
                    failed += 1
                on_progress(idx, len(selected_files), filename, result.success)

            self.root.after(0, dialog.finish, succeeded, failed)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _set_status(self, text: str):
        self._lbl_status.config(text=f"  {text}")

    def run(self):
        self.root.mainloop()
