"""FastAPI 依赖。"""

from know_answer.core.config import Settings, get_settings
from know_answer.core.db import get_db

__all__ = ["Settings", "get_settings", "get_db"]
