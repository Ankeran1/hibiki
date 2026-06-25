"""
文档扫描仪 - 批量处理模块
对目录下所有图片执行扫描并保存结果
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .utils import IMAGE_EXTENSIONS, natural_sort_key
from .pipeline import scan_file_to_dir, ScanResult


@dataclass
class BatchResult:
    """批量处理结果汇总"""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[ScanResult] = None
    output_paths: list[str] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.output_paths is None:
            self.output_paths = []


def batch_process(
    input_dir: str,
    output_dir: str,
    *,
    extensions: list[str] | None = None,
    binarize_method: str = "auto",
    canny_low: int = 30,
    canny_high: int = 100,
    progress_callback: Callable[[int, int, str, bool], None] | None = None,
) -> BatchResult:
    """
    批量处理目录下所有图片。
    progress_callback(current_1based, total, filename, success)
    """
    if ext := extensions:
        valid_exts = {e.lower().lstrip('.') for e in ext}
    else:
        valid_exts = IMAGE_EXTENSIONS

    # 收集文件
    files = sorted(
        [f for f in os.listdir(input_dir)
         if Path(f).suffix.lower() in valid_exts],
        key=natural_sort_key,
    )

    if not files:
        return BatchResult(total=0)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    result = BatchResult(total=len(files))

    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        scan_result = scan_file_to_dir(
            input_path,
            output_dir,
            binarize_method=binarize_method,
            canny_low=canny_low,
            canny_high=canny_high,
        )

        result.results.append(scan_result)

        if scan_result.success:
            result.succeeded += 1
            stem = Path(filename).stem
            result.output_paths.append(os.path.join(output_dir, f"{stem}_binarized.png"))
        else:
            result.failed += 1

        # 通知进度
        if progress_callback:
            progress_callback(i, len(files), filename, scan_result.success)

    return result
