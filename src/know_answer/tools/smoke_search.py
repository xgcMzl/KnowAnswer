"""混合检索冒烟脚本：分别跑 dense / sparse / hybrid，对比召回结果。

    python -m know_answer.tools.smoke_search --query "某个精确关键词" --top-k 5

用途：
- 确认 collection 真的带 BM25 函数（hybrid 分数应为 0.0x 量级，不是余弦值）；
- 确认关键词路能召回 dense 漏掉的精确词片段。
"""

from __future__ import annotations

import argparse
import sys

from know_answer.core.config import get_settings
from know_answer.rag import vector_store
from know_answer.rag.embedding import embed_query


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="检索问题/关键词")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--knowledge-base-id", type=int, default=None)
    return parser.parse_args(argv)


def _print_outcome(label: str, outcome) -> None:
    print(f"\n=== {label} (mode={outcome.mode}) ===")
    for rank, (node, score) in enumerate(
        zip(outcome.nodes, outcome.similarities), start=1
    ):
        metadata = node.metadata or {}
        preview = (node.get_content() or "").replace("\n", " ")[:80]
        print(
            f"{rank}. score={score:.6f} parent_id={metadata.get('parent_id')} "
            f"file={metadata.get('file_name')} text={preview}"
        )


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，文档正文里的特殊字符会导致打印失败。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)
    settings = get_settings()

    capabilities = vector_store.collection_capabilities()
    print(
        f"collection={capabilities['collection']} "
        f"sparse_field={capabilities['has_sparse_field']} "
        f"bm25_function={capabilities['has_bm25_function']} "
        f"text_analyzer={capabilities['text_analyzer_enabled']} "
        f"dim={capabilities['dim']}"
    )

    embedding = embed_query(args.query)
    common = {
        "query_str": args.query,
        "knowledge_base_id": args.knowledge_base_id,
        "top_k": args.top_k,
    }

    _print_outcome("dense", vector_store.search(embedding, mode="dense", **common))
    if settings.hybrid_retrieval_enabled:
        _print_outcome("sparse", vector_store.search(None, mode="sparse", **common))
        _print_outcome("hybrid", vector_store.search(embedding, mode="hybrid", **common))
    else:
        print("\n混合检索未开启（HYBRID_RETRIEVAL_ENABLED=false），跳过 sparse/hybrid。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
