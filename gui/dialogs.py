"""
文档扫描仪 - 对话框模块
包含批量文件选择对话框和进度对话框
"""

import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path


class BatchFileSelectDialog(tk.Toplevel):
    """批量处理文件选择对话框 — 用户勾选要处理的文件"""

    def __init__(self, parent, input_dir: str):
        super().__init__(parent)
        self.title("选择要处理的文件")
        self.geometry("450x500")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._input_dir = input_dir
        self._selected_files = []

        # 收集图片文件
        self._all_files = sorted(
            [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))],
            key=lambda s: [int(c) if c.isdigit() else c.lower()
                           for c in __import__('re').split(r'(\d+)', s)]
        )

        self._build_ui()
        self._center(parent)

    def _build_ui(self):
        tk.Label(self, text=f"文件夹: {self._input_dir}",
                 font=("微软雅黑", 9), fg="gray").pack(pady=(8, 2), anchor="w", padx=10)

        tk.Label(self, text=f"共 {len(self._all_files)} 个图片文件",
                 font=("微软雅黑", 10, "bold")).pack(pady=2)

        # 全选/取消按钮
        btn_row = tk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=3)
        tk.Button(btn_row, text="全选", command=self._select_all,
                   width=8, font=("微软雅黑", 8)).pack(side="left")
        tk.Button(btn_row, text="取消全选", command=self._deselect_all,
                   width=8, font=("微软雅黑", 8)).pack(side="left", padx=5)
        self._lbl_count = tk.Label(btn_row, text=f"已选: {len(self._all_files)}",
                                    font=("微软雅黑", 9), fg="green")
        self._lbl_count.pack(side="right")

        # 文件列表（带复选框）
        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self._listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                    selectmode="multiple",
                                    font=("Consolas", 9), height=18)
        self._listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._listbox.yview)

        # 默认全选
        for f in self._all_files:
            self._listbox.insert("end", f)
            self._listbox.selection_set("end")

        # 操作按钮
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=8)

        tk.Button(btn_frame, text="开始处理", command=self._on_start,
                   bg="#4CAF50", fg="white", font=("微软雅黑", 11, "bold"),
                   width=14).pack(side="left", padx=5)
        tk.Button(btn_frame, text="取消", command=self.destroy,
                   font=("微软雅黑", 9), width=10).pack(side="right", padx=5)

    def _select_all(self):
        self._listbox.selection_set(0, "end")
        self._update_count()

    def _deselect_all(self):
        self._listbox.selection_clear(0, "end")
        self._update_count()

    def _update_count(self):
        n = len(self._listbox.curselection())
        self._lbl_count.config(text=f"已选: {n}", fg="green" if n > 0 else "red")

    def _on_start(self):
        selected = [self._all_files[i] for i in self._listbox.curselection()]
        if not selected:
            return
        self._selected_files = selected
        self.destroy()

    def get_selected_files(self) -> list[str]:
        return self._selected_files

    def _center(self, parent):
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")


class BatchProgressDialog(tk.Toplevel):
    """批量处理进度对话框"""

    def __init__(self, parent, total: int, title: str = "批量处理"):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x380")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._total = total
        self._current = 0

        tk.Label(self, text="批量处理进度", font=("微软雅黑", 13, "bold"),
                 pady=10).pack()

        self._progress = ttk.Progressbar(self, mode="determinate",
                                          maximum=total, length=420)
        self._progress.pack(pady=5)

        self._lbl_current = tk.Label(self, text="准备中...",
                                      font=("微软雅黑", 10))
        self._lbl_current.pack(pady=3)

        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self._listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                    font=("Consolas", 9), height=10)
        self._listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._listbox.yview)

        self._lbl_summary = tk.Label(self, text="", font=("微软雅黑", 9),
                                      fg="gray")
        self._lbl_summary.pack(pady=3)

        self._btn_close = tk.Button(self, text="关闭", state="disabled",
                                     command=self.destroy, width=12,
                                     font=("微软雅黑", 10))
        self._btn_close.pack(pady=8)

        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def update_progress(self, current: int, total: int, filename: str, success: bool):
        self._current = current
        self._progress["value"] = current
        status = "✅" if success else "❌"
        self._lbl_current.config(text=f"正在处理: {filename} ({current}/{total})")
        self._listbox.insert("end", f"  {status}  {filename}")
        self._listbox.see("end")
        self._lbl_summary.config(text=f"已完成 {current}/{total}")
        self.update_idletasks()

    def finish(self, succeeded: int, failed: int):
        self._lbl_current.config(text="处理完成！")
        self._lbl_summary.config(
            text=f"完成: 成功 {succeeded} 张, 失败 {failed} 张",
            fg="green" if failed == 0 else "red"
        )
        self._btn_close.config(state="normal")
        self.update_idletasks()
