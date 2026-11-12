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
    # 混合检索需要 collection 在创建时就带 BM25 内置函数；
    # 旧的 knowanswer_chunks 没有该 schema，因此默认使用新 collection。
    milvus_collection: str = "knowanswer_chunks_v2"

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

    # 父子切片：父块只存 SQLite，子块进 Milvus
    parent_child_enabled: bool = True
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 128
    rag_parent_max_chars: int = 1600

    # 混合检索：稠密向量 + Milvus 内置 BM25，RRF 融合
    hybrid_retrieval_enabled: bool = True
    rag_rrf_k: int = 60
    # Milvus 分析器类型；中文用 chinese（jieba 分词），异常时可退回 standard
    rag_sparse_analyzer: str = "chinese"
    # RRF 融合分的绝对量级远小于余弦相似度，阈值单独配置；0 表示不过滤
    rag_hybrid_min_score: float = 0.0
    # 子块召回放大系数：先多召回子块，再按 parent_id 折叠
    rag_child_top_k_multiplier: int = 3

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
