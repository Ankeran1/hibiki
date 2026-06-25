"""
文档扫描仪 - 入口文件
"""

import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import App


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
