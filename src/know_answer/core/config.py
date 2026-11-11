"""应用配置。

所有配置项都可以通过环境变量或项目根目录下的 ``.env`` 覆盖。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """KnowAnswer 后端配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #
    app_name: str = "KnowAnswer"
    debug: bool = False
    # 逗号分隔，例如 "http://localhost:5173,http://127.0.0.1:5173"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ------------------------------------------------------------------ #
    # 存储
    # ------------------------------------------------------------------ #
    database_url: str = "sqlite:///./data/knowanswer.db"
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 20

    # ------------------------------------------------------------------ #
    # Milvus
    # ------------------------------------------------------------------ #
    milvus_uri: str = "http://106.55.165.169:19530"
    milvus_token: str = ""
    milvus_collection: str = "knowanswer_chunks"

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_cache_folder: str = r"D:\work_store\huggingface"
    embedding_device: str = "cpu"
    embedding_dim: int = 512
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    embedding_api_key: str = ""
    embedding_api_base: str = ""

    # ------------------------------------------------------------------ #
    # 对话模型
    # ------------------------------------------------------------------ #
    deepseek_api_key: str = ""
    deepseek_model_name: str = "deepseek-flash"
    deepseek_base_url: str = ""
    llm_temperature: float = 0.0

    # ------------------------------------------------------------------ #
    # 切分与检索
    # ------------------------------------------------------------------ #
    chunk_size: int = 512
    chunk_overlap: int = 64
    rag_top_k: int = 6
    rag_min_score: float = 0.0
    rag_chunk_max_chars: int = 1200
    rag_context_max_chars: int = 8000

    # ------------------------------------------------------------------ #
    # 会话记忆
    # ------------------------------------------------------------------ #
    memory_max_messages: int = 10
    memory_max_chars: int = 4000

    # ------------------------------------------------------------------ #
    # 辅助属性
    # ------------------------------------------------------------------ #
    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        """创建运行期需要的目录。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
