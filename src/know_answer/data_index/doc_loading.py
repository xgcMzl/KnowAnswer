"""基于 LlamaIndex 的文档读取、切分与 Milvus 入库模块。

支持格式
--------
- ``.pdf``  : ``llama_index.readers.file.PDFReader``（按页读取，保留 ``page_label``）
- ``.docx`` : ``llama_index.readers.file.DocxReader``（基于 ``docx2txt``）
- ``.md``   : ``llama_index.readers.file.MarkdownReader``
- ``.txt``  : ``llama_index.readers.file.FlatReader``（UTF-8；失败时按 UTF-8 容错解码）

扩展名判断大小写不敏感，切割统一使用 :class:`SentenceSplitter`。
默认 collection 为 :data:`COLLECTION_NAME`（``known_answer``），可通过
``DocumentIngestor(collection_name=...)`` 覆盖。

接口约定
--------
``DocumentIngestor.ingest(file, file_name=None)``：

* ``file`` 为 ``str`` / ``os.PathLike`` 时按**磁盘路径**处理，文件类型取自路径后缀；
  此时 ``file_name`` 可省略，若提供则仅作为 metadata 中的展示名。
* ``file`` 为 ``bytes`` / ``bytearray`` / ``memoryview`` 或带 ``read()`` 的**二进制文件对象**时，
  **必须**同时传入 ``file_name``（至少包含扩展名），因为字节流本身不携带类型信息。
  文件对象需以二进制模式（``"rb"``）打开，且从**当前读取位置**开始读取。
  内部会把字节写入同后缀的临时文件后再交给 LlamaIndex reader，读完自动清理。

Word 格式限制
-------------
``DocxReader`` 只支持 Office Open XML 格式的 ``.docx``。旧版二进制 ``.doc`` 不受支持，
请先用 LibreOffice 转换::

    soffice --headless --convert-to docx --outdir <输出目录> <文件.doc>

（macOS 上可执行文件通常为 ``/Applications/LibreOffice.app/Contents/MacOS/soffice``。）

环境变量
--------
Milvus 连接（必填 ``MILVUS_URI``）::

    MILVUS_URI    # 例如 http://localhost:19530
    MILVUS_TOKEN  # 可选，开启认证时需要（Zilliz Cloud / Milvus 鉴权）

Milvus Lite（``./milvus_demo.db``）目前主要支持 Linux/macOS；Windows 请连接
Milvus Standalone/Cluster 或 Zilliz Cloud。

嵌入模型（``EMBEDDING_PROVIDER`` 与 ``EMBEDDING_MODEL_NAME`` 必填）::

    EMBEDDING_PROVIDER     # huggingface | openai
    EMBEDDING_MODEL_NAME   # 例如 BAAI/bge-small-zh-v1.5、text-embedding-3-small
    EMBEDDING_API_KEY      # openai 时必填；亦兼容 OPENAI_API_KEY
    EMBEDDING_API_BASE     # 可选，兼容 OPENAI_API_BASE（自建/代理网关）
    EMBEDDING_CACHE_FOLDER # 可选，HuggingFace 模型缓存目录
    EMBEDDING_DEVICE       # 可选，例如 cpu / cuda
    HF_TOKEN               # 可选，huggingface_hub 会自动读取（访问 gated 模型）

代码中不包含任何密钥；未配置时会在启动阶段抛出 :class:`ConfigurationError`，
不会静默跳过写入。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, Iterator, Literal, Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode, TextNode

# 需求要求 collection 名称固定，不通过环境变量或入参覆盖。
COLLECTION_NAME: Final[str] = "known_answer"

FileType = Literal["pdf", "docx", "md", "txt"]

# 扩展名 -> 文件类型（判断时统一转小写）。
_EXTENSION_TO_FILE_TYPE: Final[dict[str, FileType]] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "md",
    ".txt": "txt",
}

# .doc 是旧版二进制格式，DocxReader 不支持，给出专门的转换提示。
_LEGACY_WORD_EXTENSIONS: Final[frozenset[str]] = frozenset({".doc"})

FileInput = Union[str, os.PathLike[str], bytes, bytearray, memoryview, BinaryIO]


# --------------------------------------------------------------------------- #
# 异常定义
# --------------------------------------------------------------------------- #
class DocumentIngestionError(Exception):
    """本模块所有可预期异常的基类，调用方可通过它统一捕获。"""


class UnsupportedFileTypeError(DocumentIngestionError):
    """文件扩展名不在支持列表中。"""


class EmptyDocumentError(DocumentIngestionError):
    """文件大小为 0，或读取/切分后没有任何有效文本。"""


class DocumentReadError(DocumentIngestionError):
    """文件不存在、入参类型不合法，或 reader 读取失败。"""


class ConfigurationError(DocumentIngestionError):
    """缺少环境变量、依赖或配置非法，或无法连接 Milvus。"""


class MilvusWriteError(DocumentIngestionError):
    """节点写入 Milvus 失败。"""


# --------------------------------------------------------------------------- #
# 配置项
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitterConfig:
    """切割参数。``chunk_size`` / ``chunk_overlap`` 以 token 计。

    开启 ``enable_parent_child`` 后做两级切分：父块只用于检索后回填上下文
    （存 SQLite），子块用于向量化与 BM25 召回（存 Milvus）。
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    paragraph_separator: str = "\n\n"
    # 句子级二次切分，包含中英文常用标点；用于避免单个超长句子撑爆 chunk。
    secondary_chunking_regex: str = r"[^,.;!?。！？；]+[,.;!?。！？；]?"

    # 父子切片
    enable_parent_child: bool = True
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 128

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ConfigurationError(f"chunk_size 必须为正整数，当前为 {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ConfigurationError(f"chunk_overlap 不能为负数，当前为 {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                f"chunk_overlap（{self.chunk_overlap}）必须小于 chunk_size（{self.chunk_size}）"
            )
        if not self.enable_parent_child:
            return
        if self.parent_chunk_size <= 0:
            raise ConfigurationError(
                f"parent_chunk_size 必须为正整数，当前为 {self.parent_chunk_size}"
            )
        if self.parent_chunk_overlap < 0:
            raise ConfigurationError(
                f"parent_chunk_overlap 不能为负数，当前为 {self.parent_chunk_overlap}"
            )
        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ConfigurationError(
                "parent_chunk_overlap"
                f"（{self.parent_chunk_overlap}）必须小于 parent_chunk_size"
                f"（{self.parent_chunk_size}）"
            )
        if self.parent_chunk_size < self.chunk_size:
            raise ConfigurationError(
                f"parent_chunk_size（{self.parent_chunk_size}）不能小于"
                f" chunk_size（{self.chunk_size}）"
            )


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接配置。``token`` 允许为空（本地/无鉴权实例）。"""

    uri: str
    token: str = ""

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        uri = os.getenv("MILVUS_URI", "http://106.55.165.169:19530").strip()
        if not uri:
            raise ConfigurationError(
                "未配置 Milvus 连接：请设置环境变量 MILVUS_URI，"
                "例如 http://localhost:19530（Windows 建议连接 Milvus 服务；"
                "Milvus Lite 的 ./milvus_demo.db 主要用于 Linux/macOS）。"
            )
        token = os.getenv("MILVUS_TOKEN", "").strip()
        return cls(uri=uri, token=token)


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入模型配置。API Key 只从环境变量/构造参数读取，代码中不落盘。"""

    provider: Literal["huggingface", "openai"]
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    cache_folder: Optional[str] = None
    device: Optional[str] = None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        raw_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
        if not raw_provider:
            raise ConfigurationError(
                "未配置嵌入模型：请设置环境变量 EMBEDDING_PROVIDER（huggingface 或 openai）。"
            )
        if raw_provider in {"huggingface", "hf"}:
            provider: Literal["huggingface", "openai"] = "huggingface"
        elif raw_provider == "openai":
            provider = "openai"
        else:
            raise ConfigurationError(
                f"不支持的 EMBEDDING_PROVIDER：'{raw_provider}'，仅支持 huggingface / openai。"
            )

        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5").strip()
        if not model_name:
            raise ConfigurationError(
                "未配置嵌入模型名称：请设置环境变量 EMBEDDING_MODEL_NAME，"
                "例如 BAAI/bge-small-zh-v1.5 或 text-embedding-3-small。"
            )

        api_key = (
            os.getenv("EMBEDDING_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
            or None
        )
        api_base = (
            os.getenv("EMBEDDING_API_BASE", "").strip()
            or os.getenv("OPENAI_API_BASE", "").strip()
            or None
        )
        if provider == "openai" and not api_key:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=openai 时必须设置 EMBEDDING_API_KEY（或 OPENAI_API_KEY）。"
            )

        return cls(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            cache_folder=os.getenv("EMBEDDING_CACHE_FOLDER", "D:\\work_store\\huggingface").strip() or None,
            device=os.getenv("EMBEDDING_DEVICE", "cpu").strip() or None,
        )


def build_embed_model(config: EmbeddingConfig) -> BaseEmbedding:
    """根据配置构建 LlamaIndex 嵌入模型。

    依赖缺失、模型初始化失败都会抛出 :class:`ConfigurationError`，不会静默降级。
    """
    if config.provider == "huggingface":
        try:
            # 版本敏感：HuggingFaceEmbedding 位于 llama-index-embeddings-huggingface。
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:
            raise ConfigurationError(
                "缺少依赖 llama-index-embeddings-huggingface，"
                "请执行：uv add llama-index-embeddings-huggingface"
            ) from exc

        kwargs: dict[str, object] = {"model_name": config.model_name}
        if config.cache_folder:
            kwargs["cache_folder"] = config.cache_folder
        if config.device:
            kwargs["device"] = config.device
        try:
            return HuggingFaceEmbedding(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            raise ConfigurationError(
                f"初始化 HuggingFace 嵌入模型 '{config.model_name}' 失败：{exc}"
            ) from exc

    # provider == "openai"
    try:
        # 版本敏感：OpenAIEmbedding 位于 llama-index-embeddings-openai，需额外安装。
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:
        raise ConfigurationError(
            "缺少依赖 llama-index-embeddings-openai，"
            "请执行：uv add llama-index-embeddings-openai"
        ) from exc

    if not config.api_key:
        raise ConfigurationError(
            "EMBEDDING_PROVIDER=openai 时必须设置 EMBEDDING_API_KEY（或 OPENAI_API_KEY）。"
        )

    # 版本敏感：不同版本中 api_key / api_base 的参数名可能变化
    # （旧版本通常直接从 OPENAI_API_KEY / OPENAI_API_BASE 读取环境变量）。
    try:
        return OpenAIEmbedding(
            model=config.model_name,
            api_key=config.api_key,
            api_base=config.api_base,
        )
    except Exception as exc:
        raise ConfigurationError(
            f"初始化 OpenAI 嵌入模型 '{config.model_name}' 失败：{exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# 读取 / 切分辅助
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _MaterializedInput:
    """统一的文件输入视图：路径 + 类型判断名 + 展示名 + 来源类型。"""

    path: Path
    type_name: str
    display_name: str
    source_type: Literal["path", "bytes"]
    source_path: Optional[str]


@dataclass(frozen=True)
class ParentChunkDraft:
    """父块草稿：由切分阶段产出，交由调用方落 SQLite（不进 Milvus）。"""

    parent_id: str
    parent_index: int
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SplitOutcome:
    """两级切分结果：子块进向量库，父块另行持久化。"""

    child_nodes: list[BaseNode]
    parent_chunks: tuple[ParentChunkDraft, ...] = ()


@dataclass(frozen=True)
class IngestResult:
    """一次入库的结果，便于调用方与验证脚本断言。

    ``node_count`` / ``node_ids`` 指写入 Milvus 的**子块**；父块只随
    ``parent_chunks`` 返回，由调用方决定持久化方式。
    """

    file_name: str
    file_type: FileType
    node_count: int
    node_ids: tuple[str, ...]
    collection_name: str = COLLECTION_NAME
    parent_count: int = 0
    parent_chunks: tuple[ParentChunkDraft, ...] = ()


def _read_binary_input(file: FileInput) -> bytes:
    if isinstance(file, (bytes, bytearray, memoryview)):
        return bytes(file)

    if hasattr(file, "read"):
        data = file.read()  # 从当前读取位置开始，不主动 seek
        if isinstance(data, str):
            raise DocumentReadError(
                "文件对象需以二进制模式打开（例如 open(path, 'rb')），"
                "当前 read() 返回了 str。"
            )
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise DocumentReadError(
                f"文件对象 read() 返回了不支持的类型：{type(data).__name__}，需返回 bytes。"
            )
        return bytes(data)

    raise DocumentReadError(
        f"不支持的 file 参数类型：{type(file).__name__}。"
        "支持 str/os.PathLike 路径、bytes/bytearray/memoryview，或带 read() 的二进制文件对象。"
    )


@contextmanager
def _materialize_input(file: FileInput, file_name: Optional[str]) -> Iterator[_MaterializedInput]:
    """把路径或字节流统一成磁盘路径；字节流使用同后缀临时文件。"""
    if isinstance(file, (str, os.PathLike)):
        path = Path(file)
        if not path.exists():
            raise DocumentReadError(f"文件不存在：{path}")
        if not path.is_file():
            raise DocumentReadError(f"路径不是文件：{path}")
        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"文件为空（0 字节）：{path}")

        display_name = (file_name or "").strip() or path.name
        yield _MaterializedInput(
            path=path,
            # 路径输入时类型始终以路径后缀为准，避免 file_name 覆盖导致误判。
            type_name=path.name,
            display_name=display_name,
            source_type="path",
            source_path=str(path.resolve()),
        )
        return

    data = _read_binary_input(file)
    if not file_name or not file_name.strip():
        raise DocumentReadError(
            "传入 bytes 或文件对象时必须提供 file_name（至少包含扩展名），"
            "例如 file_name='report.pdf'，字节流本身不携带文件类型。"
        )
    display_name = Path(file_name.strip()).name
    suffix = Path(display_name).suffix.lower()
    if not suffix:
        raise DocumentReadError(
            f"file_name 缺少扩展名，无法判断文件类型：'{display_name}'。"
        )
    if not data:
        raise EmptyDocumentError(f"文件内容为空（0 字节）：{display_name}")

    with tempfile.TemporaryDirectory(prefix="knowanswer_ingest_") as tmp_dir:
        temp_path = Path(tmp_dir) / f"upload{suffix}"
        temp_path.write_bytes(data)
        yield _MaterializedInput(
            path=temp_path,
            type_name=display_name,
            display_name=display_name,
            source_type="bytes",
            source_path=None,
        )


def _detect_file_type(type_name: str) -> FileType:
    """按扩展名（大小写不敏感）判断文件类型。"""
    suffix = Path(type_name).suffix.lower()

    if suffix in _LEGACY_WORD_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"不支持旧版 Word 格式 '{suffix}'（文件：{type_name}）。"
            "LlamaIndex 的 DocxReader 仅支持 OOXML 的 .docx；"
            "请先转换：soffice --headless --convert-to docx --outdir <输出目录> <文件.doc>"
        )

    try:
        return _EXTENSION_TO_FILE_TYPE[suffix]
    except KeyError:
        supported = ", ".join(sorted(_EXTENSION_TO_FILE_TYPE))
        shown = suffix or "<无扩展名>"
        raise UnsupportedFileTypeError(
            f"不支持的文件类型 '{shown}'（文件：{type_name}）。当前仅支持：{supported}"
        ) from None


def _build_reader(file_type: FileType):
    """按文件类型选择 LlamaIndex reader。"""
    try:
        # 版本敏感：reader 位于独立包 llama-index-readers-file（>=0.7）。
        from llama_index.readers.file import (
            DocxReader,
            FlatReader,
            MarkdownReader,
            PDFReader,
        )
    except ImportError as exc:
        raise ConfigurationError(
            "缺少依赖 llama-index-readers-file，"
            "请执行：uv add llama-index-readers-file docx2txt"
        ) from exc

    if file_type == "pdf":
        # 版本敏感：return_full_document=False 表示每一页生成一个 Document，
        # metadata 中保留 page_label，便于后续按页引用。
        return PDFReader(return_full_document=False)
    if file_type == "docx":
        return DocxReader()
    if file_type == "txt":
        return FlatReader()
    return MarkdownReader()


def _load_documents(
    materialized: _MaterializedInput,
    file_type: FileType,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """调用 reader 读取文件，并补充来源 metadata。"""
    reader = _build_reader(file_type)
    extra_info: dict[str, object] = {
        "file_name": materialized.display_name,
        "file_type": file_type,
        "source_type": materialized.source_type,
    }
    if materialized.source_path is not None:
        extra_info["source_path"] = materialized.source_path
    if extra_metadata:
        extra_info.update(extra_metadata)

    try:
        if file_type == "md":
            # MarkdownReader.load_data 的 file 参数为 str。
            documents = reader.load_data(
                file=str(materialized.path), extra_info=extra_info
            )
        else:
            documents = reader.load_data(
                file=materialized.path, extra_info=extra_info
            )
    except UnicodeDecodeError as exc:
        # FlatReader 使用严格 UTF-8；Windows 上常见 GBK 文本需要兜底。
        if file_type != "txt":
            raise DocumentReadError(
                f"读取文件 '{materialized.display_name}' 失败（编码错误）：{exc}"
            ) from exc
        text = materialized.path.read_text(encoding="utf-8", errors="replace")
        documents = [Document(text=text, metadata=dict(extra_info))]
    except Exception as exc:
        raise DocumentReadError(
            f"读取文件 '{materialized.display_name}' 失败（类型：{file_type}）：{exc}"
        ) from exc

    if not documents:
        raise EmptyDocumentError(
            f"读取文件 '{materialized.display_name}' 未返回任何文档，文件可能已损坏或为空。"
        )

    # reader 实现可能覆盖 metadata；这里确保关键字段一定存在。
    for document in documents:
        document.metadata.setdefault("file_name", materialized.display_name)
        document.metadata.setdefault("file_type", file_type)
    return documents


def _resolve_file_key(
    materialized: _MaterializedInput,
    extra_metadata: Optional[dict[str, Any]],
) -> str:
    """父块 ID 前缀：业务链路优先用 ``file_id``，独立使用时退回稳定哈希。"""
    if extra_metadata:
        file_id = extra_metadata.get("file_id")
        if file_id is not None:
            return str(file_id)

    seed = f"{materialized.display_name}|{materialized.source_path or ''}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
class DocumentIngestor:
    """读取文件 -> 切分节点 -> 生成 embedding -> 写入 Milvus。"""

    def __init__(
        self,
        *,
        milvus_config: Optional[MilvusConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        splitter_config: Optional[SplitterConfig] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[Any] = None,
        collection_name: str = COLLECTION_NAME,
        similarity_metric: str = "COSINE",
        scalar_field_names: Optional[list[str]] = None,
        scalar_field_types: Optional[list[Any]] = None,
        overwrite: bool = False,
        show_progress: bool = False,
    ) -> None:
        """
        Args:
            milvus_config: 缺省时从 ``MILVUS_URI`` / ``MILVUS_TOKEN`` 读取。
            embedding_config: 缺省时从 ``EMBEDDING_*`` 环境变量读取。
            splitter_config: 切割参数，缺省使用 :class:`SplitterConfig` 的默认值。
            embed_model: 可直接注入已构建的 LlamaIndex 嵌入模型，便于测试或复用。
            vector_store: 可直接注入已构建的 Milvus 适配器；注入后入库与检索共用实例。
            collection_name: Milvus collection 名称，默认 ``known_answer``。
            similarity_metric: 向量相似度度量，默认 ``COSINE``。
            scalar_field_names / scalar_field_types: 新建 collection 时声明的业务
                标量字段（例如 ``knowledge_base_id`` / ``file_id`` / ``file_name``）。
            overwrite: 是否在首次建连时 drop 并重建 collection。
                **会删除已有数据**，默认 ``False``；只建议在单文件全量重建时使用。
            show_progress: 生成 embedding 时是否显示进度条。
        """
        self._milvus_config = milvus_config or MilvusConfig.from_env()
        self._splitter_config = splitter_config or SplitterConfig()
        self._splitter_config.validate()

        if embed_model is not None:
            self._embed_model: Optional[BaseEmbedding] = embed_model
            self._embedding_config: Optional[EmbeddingConfig] = embedding_config
        else:
            self._embedding_config = embedding_config or EmbeddingConfig.from_env()
            self._embed_model = None

        self._collection_name = collection_name
        self._similarity_metric = similarity_metric
        self._scalar_field_names = scalar_field_names
        self._scalar_field_types = scalar_field_types
        self._injected_vector_store = vector_store

        self._overwrite = overwrite
        self._show_progress = show_progress
        self._vector_store = vector_store
        self._store_dim: Optional[int] = None

    # ----------------------------- 公开方法 ----------------------------- #
    def ingest(
        self,
        file: FileInput,
        *,
        file_name: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> IngestResult:
        """读取单个文件并写入 Milvus。

        Args:
            file: 文件路径，或 bytes / 二进制文件对象。
            file_name: 仅当 ``file`` 为字节流时必填；带扩展名用于类型判断，
                也作为 metadata 中的 ``file_name``。
            extra_metadata: 附加到每个节点 metadata 的业务字段；用于声明了
                标量字段的 collection（如 ``knowledge_base_id`` / ``file_id``）。

        Returns:
            :class:`IngestResult`，包含写入节点数量与节点 ID。

        Raises:
            DocumentIngestionError: 及其子类，涵盖类型不支持、空文件、读取失败、
                配置缺失、Milvus 连接或写入失败。
        """
        with _materialize_input(file, file_name) as materialized:
            file_type = _detect_file_type(materialized.type_name)
            documents = _load_documents(materialized, file_type, extra_metadata)

            if all(not (document.text or "").strip() for document in documents):
                raise EmptyDocumentError(
                    f"文件 '{materialized.display_name}' 未解析出任何文本。"
                    "若为扫描版 PDF，请先做 OCR；若文件为空或损坏，请检查源文件。"
                )

            split = self._split(
                documents,
                file_key=_resolve_file_key(materialized, extra_metadata),
            )
            nodes = split.child_nodes
            if not nodes:
                raise EmptyDocumentError(
                    f"文件 '{materialized.display_name}' 切分后没有有效节点。"
                )

            embeddings = self._embed_nodes(nodes)
            dim = len(embeddings[0])
            vector_store = self._get_vector_store(dim)

            try:
                # 版本敏感：force_flush 通过 add(**add_kwargs) 传入，
                # 使数据尽快对后续查询可见；旧版本若不支持会被忽略或报错。
                node_ids = vector_store.add(nodes, force_flush=True)
            except Exception as exc:
                raise MilvusWriteError(
                    f"写入 Milvus collection '{self._collection_name}' 失败：{exc}。"
                    "请检查：1) MILVUS_URI / MILVUS_TOKEN 是否有效；"
                    "2) 若 collection 已存在，其 embedding 维度是否与当前模型一致；"
                    "3) 必要时使用 overwrite=True 重建（会删除已有数据）。"
                ) from exc

            if not isinstance(node_ids, list) or len(node_ids) != len(nodes):
                raise MilvusWriteError(
                    f"Milvus 返回的写入 ID 数量异常：节点数 {len(nodes)}，"
                    f"返回 {len(node_ids) if node_ids is not None else 'None'}。"
                )

            return IngestResult(
                file_name=materialized.display_name,
                file_type=file_type,
                node_count=len(nodes),
                node_ids=tuple(node_ids),
                collection_name=self._collection_name,
                parent_count=len(split.parent_chunks),
                parent_chunks=split.parent_chunks,
            )

    # ----------------------------- 内部方法 ----------------------------- #
    def _split(self, documents: list[Document], *, file_key: str) -> SplitOutcome:
        """把文档切成「子块（进 Milvus）+ 父块（进 SQLite）」。

        - 父块不跨 Document（PDF 每页独立），避免 page_label 串页。
        - ``parent_index`` / ``chunk_index`` 均按整份文件全局递增。
        - 关闭父子切片时退化为原来的扁平切分，``parent_chunks`` 为空。
        """
        splitter_config = self._splitter_config
        child_splitter = SentenceSplitter(
            chunk_size=splitter_config.chunk_size,
            chunk_overlap=splitter_config.chunk_overlap,
            paragraph_separator=splitter_config.paragraph_separator,
            secondary_chunking_regex=splitter_config.secondary_chunking_regex,
        )

        if not splitter_config.enable_parent_child:
            nodes: list[BaseNode] = []
            for document in documents:
                # 只对正文调用 split_text，chunk_size 不会因 metadata 较长而失效
                # （get_nodes_from_documents 会把 metadata 计入 token 预算，
                #   小 chunk_size 时会抛 "Metadata length is longer than chunk size"）。
                for text in child_splitter.split_text(document.text):
                    if not text.strip():
                        continue
                    nodes.append(self._make_node(text, document))
            for index, node in enumerate(nodes):
                node.metadata["chunk_index"] = index
            return SplitOutcome(child_nodes=nodes)

        parent_splitter = SentenceSplitter(
            chunk_size=splitter_config.parent_chunk_size,
            chunk_overlap=splitter_config.parent_chunk_overlap,
            paragraph_separator=splitter_config.paragraph_separator,
            secondary_chunking_regex=splitter_config.secondary_chunking_regex,
        )

        children: list[BaseNode] = []
        parents: list[ParentChunkDraft] = []
        parent_index = 0
        child_index = 0

        for document in documents:
            for parent_text in parent_splitter.split_text(document.text):
                if not parent_text.strip():
                    continue

                parent_meta = dict(document.metadata)
                parent_meta["parent_id"] = f"{file_key}:{parent_index}"
                parent_meta["parent_index"] = parent_index
                parents.append(
                    ParentChunkDraft(
                        parent_id=parent_meta["parent_id"],
                        parent_index=parent_index,
                        text=parent_text,
                        metadata=parent_meta,
                    )
                )

                for local_index, child_text in enumerate(
                    child_splitter.split_text(parent_text)
                ):
                    if not child_text.strip():
                        continue
                    node = self._make_node(child_text, document)
                    node.metadata["parent_id"] = parent_meta["parent_id"]
                    node.metadata["parent_index"] = parent_index
                    node.metadata["child_index_in_parent"] = local_index
                    node.metadata["chunk_index"] = child_index
                    children.append(node)
                    child_index += 1

                parent_index += 1

        return SplitOutcome(child_nodes=children, parent_chunks=tuple(parents))

    @staticmethod
    def _make_node(text: str, document: Document) -> BaseNode:
        return TextNode(
            text=text,
            # 每个节点使用独立的 metadata 副本。
            metadata=dict(document.metadata),
            excluded_embed_metadata_keys=list(document.excluded_embed_metadata_keys),
            excluded_llm_metadata_keys=list(document.excluded_llm_metadata_keys),
        )

    def _embed_nodes(self, nodes: list[BaseNode]) -> list[list[float]]:
        embed_model = self._get_embed_model()
        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        try:
            embeddings = embed_model.get_text_embedding_batch(
                texts, show_progress=self._show_progress
            )
        except Exception as exc:
            raise ConfigurationError(
                f"生成 embedding 失败（模型：{self._embedding_config.model_name if self._embedding_config else '自定义 embed_model'}）：{exc}"
            ) from exc

        if len(embeddings) != len(nodes):
            raise ConfigurationError(
                f"embedding 数量与节点数量不一致：{len(embeddings)} != {len(nodes)}。"
            )
        if not embeddings or not embeddings[0]:
            raise ConfigurationError("嵌入模型返回了空向量，请检查模型配置。")

        dims = {len(vector) for vector in embeddings}
        if len(dims) != 1:
            raise ConfigurationError(f"embedding 维度不一致：{sorted(dims)}。")

        for node, vector in zip(nodes, embeddings):
            node.embedding = vector
        return embeddings

    def _get_embed_model(self) -> BaseEmbedding:
        if self._embed_model is None:
            if self._embedding_config is None:  # 防御性检查，正常构造不会走到这里
                raise ConfigurationError("未配置嵌入模型。")
            self._embed_model = build_embed_model(self._embedding_config)
        return self._embed_model

    def _get_vector_store(self, dim: int):
        # 外部注入的实例由调用方负责维度一致性，这里直接复用。
        if self._injected_vector_store is not None:
            return self._injected_vector_store

        if self._vector_store is not None:
            if self._store_dim != dim:
                raise ConfigurationError(
                    f"本次 embedding 维度为 {dim}，与已建连接的 {self._store_dim} 不一致，"
                    "请为新模型使用新的 DocumentIngestor 实例。"
                )
            return self._vector_store

        try:
            # 版本敏感：llama-index-vector-stores-milvus 的参数名随版本演进，
            # 以下基于 1.2.x；升级前请核对 MilvusVectorStore.__init__ 签名。
            from llama_index.vector_stores.milvus import MilvusVectorStore
        except ImportError as exc:
            raise ConfigurationError(
                "缺少依赖 llama-index-vector-stores-milvus，"
                "请执行：uv add llama-index-vector-stores-milvus"
            ) from exc

        store_kwargs: dict[str, Any] = {
            "uri": self._milvus_config.uri,
            "token": self._milvus_config.token,
            "collection_name": self._collection_name,
            "dim": dim,
            "overwrite": self._overwrite,
            "enable_dense": True,
            "enable_sparse": False,
            "similarity_metric": self._similarity_metric,
        }
        if self._scalar_field_names and self._scalar_field_types:
            store_kwargs["scalar_field_names"] = list(self._scalar_field_names)
            store_kwargs["scalar_field_types"] = list(self._scalar_field_types)

        try:
            vector_store = MilvusVectorStore(**store_kwargs)
        except Exception as exc:
            raise ConfigurationError(
                f"连接 Milvus 或创建 collection '{self._collection_name}' 失败：{exc}。"
                "请检查环境变量 MILVUS_URI / MILVUS_TOKEN，以及 Milvus 服务是否可访问。"
            ) from exc

        self._vector_store = vector_store
        self._store_dim = dim
        return vector_store


# --------------------------------------------------------------------------- #
# 命令行示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        raise SystemExit(
            "用法：python -m know_answer.data_index.doc_loading <文件1> [文件2 ...]"
        )

    ingestor = DocumentIngestor()
    has_error = False
    for raw_path in sys.argv[1:]:
        try:
            result = ingestor.ingest(raw_path)
            print(
                f"[OK] {result.file_name} -> {result.node_count} 个节点，"
                f"collection={result.collection_name}"
            )
        except DocumentIngestionError as exc:
            has_error = True
            print(f"[FAIL] {raw_path}: {exc}", file=sys.stderr)

    raise SystemExit(1 if has_error else 0)
