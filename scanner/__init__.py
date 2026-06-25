"""
文档扫描仪 - 核心模块
"""

from .pipeline import scan_image, scan_file, scan_file_to_dir, ScanResult
from .detection import detect_document, detect_and_draw
from .batch import batch_process, BatchResult
from .screen_detect import detect_bright_screen
from .edge_refine import refine_corners

__all__ = [
    "scan_image", "scan_file", "scan_file_to_dir", "ScanResult",
    "detect_document", "detect_and_draw",
    "batch_process", "BatchResult",
    "detect_bright_screen", "refine_corners",
]
