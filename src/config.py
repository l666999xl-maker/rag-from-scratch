"""
确认配置环境
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录（src 的上一级）
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

# 对话模型
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = os.getenv("MODEL", "deepseek-v4.1-flash-1017")

# 向量模型
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 目录
PDF_DIR = ROOT / "data" / "raw" / "pdf"
BUILD_DIR = ROOT / "build"


def check() -> list[str]:
    """检查必需的配置项，返回缺失项列表。"""
    required = {
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "DASHSCOPE_API_KEY": DASHSCOPE_API_KEY,
        "EMBEDDING_BASE_URL": EMBEDDING_BASE_URL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
    }
    return [name for name, value in required.items() if not value]