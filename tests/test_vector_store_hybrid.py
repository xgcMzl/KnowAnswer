import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.vector_stores.types import VectorStoreQueryMode

import know_answer.rag.vector_store as vector_store
from know_answer.core.errors import MilvusUnavailableError


def _schema(*, sparse: bool = True) -> dict:
    fields = [
        {"name": "id", "params": {"max_length": 65535}},
        {"name": "doc_id", "params": {"max_length": 65535}},
        {
            "name": "text",
            "params": {"max_length": 65535, "enable_analyzer": sparse},
        },
        {"name": "knowledge_base_id", "params": {}},
        {"name": "file_id", "params": {}},
        {"name": "file_name", "params": {"max_length": 65535}},
        {"name": "embedding", "params": {"dim": 512}},
    ]
    functions = []
    if sparse:
        fields.append({"name": "sparse_embedding", "params": {}})
        functions.append(
            {
                "type": "BM25",
                "input_field_names": ["text"],
                "output_field_names": ["sparse_embedding"],
            }
        )
    return {"fields": fields, "functions": functions}


class FakeClient:
    def __init__(self, info: dict) -> None:
        self._info = info

    def describe_collection(self, name: str) -> dict:
        return self._info


class FakeMilvusVectorStore:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.client = FakeClient(_schema())

    def query(self, query, **kwargs):
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


class LegacyMilvusVectorStore(FakeMilvusVectorStore):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.client = FakeClient(_schema(sparse=False))


class RecordingStore:
    def __init__(self, *, fail_hybrid: bool = False) -> None:
        self.queries: list[VectorStoreQuery] = []
        self.fail_hybrid = fail_hybrid

    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        self.queries.append(query)
        if self.fail_hybrid and query.mode == VectorStoreQueryMode.HYBRID:
            raise ValueError("server does not support hybrid search")
        return VectorStoreQueryResult(
            nodes=[TextNode(text="片段")], similarities=[0.03], ids=["node-1"]
        )


@pytest.fixture(autouse=True)
def _reset_singleton():
    vector_store.reset_vector_store()
    yield
    vector_store.reset_vector_store()


def _patch_store_class(monkeypatch, klass) -> None:
    monkeypatch.setattr(
        "llama_index.vector_stores.milvus.MilvusVectorStore", klass
    )


def test_build_enables_bm25_and_rrf(monkeypatch) -> None:
    _patch_store_class(monkeypatch, FakeMilvusVectorStore)

    store = vector_store.get_vector_store()
    kwargs = store.kwargs

    assert kwargs["enable_sparse"] is True
    assert kwargs["hybrid_ranker"] == "RRFRanker"
    assert kwargs["hybrid_ranker_params"] == {"k": 60}
    assert kwargs["sparse_embedding_function"].analyzer_params == {"type": "chinese"}


def test_legacy_collection_is_rejected(monkeypatch) -> None:
    _patch_store_class(monkeypatch, LegacyMilvusVectorStore)

    with pytest.raises(MilvusUnavailableError) as excinfo:
        vector_store.get_vector_store()

    message = str(excinfo.value)
    assert "不支持混合检索" in message
    assert "knowanswer_chunks_v2" in message


def test_search_builds_hybrid_query(monkeypatch) -> None:
    store = RecordingStore()
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: store)

    outcome = vector_store.search(
        [0.1] * 8,
        query_str="问题",
        knowledge_base_id=3,
        top_k=4,
    )

    assert outcome.mode == "hybrid"
    query = store.queries[0]
    assert query.mode == VectorStoreQueryMode.HYBRID
    assert query.query_str == "问题"
    assert query.query_embedding == [0.1] * 8
    assert query.similarity_top_k == 4
    assert query.filters is not None


def test_hybrid_falls_back_to_dense(monkeypatch) -> None:
    store = RecordingStore(fail_hybrid=True)
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: store)

    outcome = vector_store.search(
        [0.1] * 8, query_str="问题", knowledge_base_id=None, top_k=2
    )

    assert outcome.mode == "dense"
    assert [query.mode for query in store.queries] == [
        VectorStoreQueryMode.HYBRID,
        VectorStoreQueryMode.DEFAULT,
    ]


def test_health_reports_hybrid_capabilities(monkeypatch) -> None:
    store = FakeMilvusVectorStore()
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: store)

    report = vector_store.health()

    assert report["status"] == "ok"
    assert report["hybrid_enabled"] is True
    assert report["sparse_field"] is True
    assert report["bm25_function"] is True
