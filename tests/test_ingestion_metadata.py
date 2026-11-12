from pathlib import Path

from know_answer.data_index.doc_loading import (
    DocumentIngestor,
    EmbeddingConfig,
    MilvusConfig,
    SplitterConfig,
    _detect_file_type,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.nodes = []

    def add(self, nodes, **kwargs):
        self.nodes.extend(nodes)
        return [node.node_id for node in nodes]


def test_detect_txt_file_type() -> None:
    assert _detect_file_type("demo.txt") == "txt"
    assert _detect_file_type("DEMO.TXT") == "txt"


def test_ingest_injects_business_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("第一段内容。\n\n第二段内容。", encoding="utf-8")

    store = FakeVectorStore()
    ingestor = DocumentIngestor(
        milvus_config=MilvusConfig(uri="http://localhost:19530"),
        embedding_config=EmbeddingConfig(
            provider="huggingface", model_name="fake-model"
        ),
        splitter_config=SplitterConfig(chunk_size=64, chunk_overlap=0),
        vector_store=store,
        collection_name="test_collection",
    )

    def fake_embed(self, nodes):
        vectors = [[0.1] * 8 for _ in nodes]
        for node, vector in zip(nodes, vectors):
            node.embedding = vector
        return vectors

    monkeypatch.setattr(DocumentIngestor, "_embed_nodes", fake_embed)

    result = ingestor.ingest(
        str(path),
        extra_metadata={
            "knowledge_base_id": 7,
            "file_id": 42,
            "file_name": "renamed.txt",
        },
    )

    assert result.collection_name == "test_collection"
    assert result.node_count >= 1
    assert len(store.nodes) == result.node_count

    metadata = store.nodes[0].metadata
    assert metadata["knowledge_base_id"] == 7
    assert metadata["file_id"] == 42
    assert metadata["file_name"] == "renamed.txt"
    assert metadata["file_type"] == "txt"
    assert "chunk_index" in metadata

    # 父子切片：子块带 parent_id，父块只随结果返回、不进向量库
    assert result.parent_count >= 1
    assert metadata["parent_id"] == f"42:{metadata['parent_index']}"
    assert len(result.parent_chunks) == result.parent_count
    assert all("parent_id" in node.metadata for node in store.nodes)
    assert any(
        store.nodes[0].get_content() in draft.text for draft in result.parent_chunks
    )


def test_ingest_without_parent_child_returns_flat_nodes(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "flat.txt"
    path.write_text("第一段内容。\n\n第二段内容。", encoding="utf-8")

    store = FakeVectorStore()
    ingestor = DocumentIngestor(
        milvus_config=MilvusConfig(uri="http://localhost:19530"),
        embedding_config=EmbeddingConfig(
            provider="huggingface", model_name="fake-model"
        ),
        splitter_config=SplitterConfig(
            chunk_size=64, chunk_overlap=0, enable_parent_child=False
        ),
        vector_store=store,
        collection_name="test_collection",
    )
    monkeypatch.setattr(
        DocumentIngestor,
        "_embed_nodes",
        lambda self, nodes: [[0.1] * 8 for _ in nodes],
    )

    result = ingestor.ingest(
        str(path),
        extra_metadata={"knowledge_base_id": 7, "file_id": 42, "file_name": "flat.txt"},
    )

    assert result.parent_count == 0
    assert result.parent_chunks == ()
    assert all("parent_id" not in node.metadata for node in store.nodes)
