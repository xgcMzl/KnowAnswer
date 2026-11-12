import pytest
from llama_index.core.schema import Document

from know_answer.data_index.doc_loading import (
    ConfigurationError,
    DocumentIngestor,
    MilvusConfig,
    SplitterConfig,
)


def _ingestor(**splitter_kwargs) -> DocumentIngestor:
    return DocumentIngestor(
        milvus_config=MilvusConfig(uri="http://localhost:19530"),
        splitter_config=SplitterConfig(**splitter_kwargs),
        collection_name="test_collection",
    )


def _documents() -> list[Document]:
    return [
        Document(text="第一页内容。" * 20, metadata={"page_label": "1", "file_id": 42}),
        Document(text="第二页内容。" * 20, metadata={"page_label": "2", "file_id": 42}),
    ]


def test_parent_index_is_global_across_documents() -> None:
    ingestor = _ingestor(
        chunk_size=16, chunk_overlap=0, parent_chunk_size=32, parent_chunk_overlap=0
    )

    outcome = ingestor._split(_documents(), file_key="42")

    assert outcome.parent_chunks
    assert [item.parent_index for item in outcome.parent_chunks] == list(
        range(len(outcome.parent_chunks))
    )
    assert len({item.parent_id for item in outcome.parent_chunks}) == len(
        outcome.parent_chunks
    )


def test_parent_id_format_and_child_metadata() -> None:
    ingestor = _ingestor(
        chunk_size=16, chunk_overlap=0, parent_chunk_size=32, parent_chunk_overlap=0
    )

    outcome = ingestor._split(_documents(), file_key="42")

    parent_ids = {item.parent_id for item in outcome.parent_chunks}
    assert all(pid.startswith("42:") for pid in parent_ids)

    for index, node in enumerate(outcome.child_nodes):
        assert node.metadata["chunk_index"] == index
        assert node.metadata["parent_id"] in parent_ids
        assert "parent_index" in node.metadata
        assert "child_index_in_parent" in node.metadata


def test_parent_text_contains_its_children() -> None:
    ingestor = _ingestor(
        chunk_size=16, chunk_overlap=0, parent_chunk_size=32, parent_chunk_overlap=0
    )

    outcome = ingestor._split(_documents(), file_key="42")
    parents = {item.parent_id: item for item in outcome.parent_chunks}

    for node in outcome.child_nodes:
        assert node.get_content() in parents[node.metadata["parent_id"]].text


def test_parent_child_disabled_returns_flat_nodes() -> None:
    ingestor = _ingestor(chunk_size=16, chunk_overlap=0, enable_parent_child=False)

    outcome = ingestor._split(_documents(), file_key="42")

    assert outcome.parent_chunks == ()
    assert outcome.child_nodes
    assert all("parent_id" not in node.metadata for node in outcome.child_nodes)
    assert [node.metadata["chunk_index"] for node in outcome.child_nodes] == list(
        range(len(outcome.child_nodes))
    )


def test_splitter_config_rejects_invalid_parent_size() -> None:
    with pytest.raises(ConfigurationError):
        SplitterConfig(
            chunk_size=64,
            parent_chunk_size=32,
        ).validate()

    with pytest.raises(ConfigurationError):
        SplitterConfig(
            chunk_size=64,
            parent_chunk_size=128,
            parent_chunk_overlap=128,
        ).validate()
