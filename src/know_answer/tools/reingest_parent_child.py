"""把已入库文件重新导入到父子切片 + 混合检索链路。

典型用法（迁移到新的 collection）：

    python -m know_answer.tools.reingest_parent_child --dry-run
    python -m know_answer.tools.reingest_parent_child
    python -m know_answer.tools.reingest_parent_child --include-failed

注意：迁移期间请停止上传与其他入库任务，避免与后台 ``process_file`` 并发。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

from know_answer.core.config import get_settings
from know_answer.core.db import session_scope
from know_answer.db.models import FileStatus, KnowledgeFile
from know_answer.rag import vector_store
from know_answer.services import ingest_service

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_storage_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将要处理的文件，不执行入库",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="同时重新入库 failed / processing 的记录（默认只处理 ready）",
    )
    parser.add_argument(
        "--file-id",
        type=int,
        action="append",
        default=None,
        help="只处理指定文件 ID，可重复传入",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，文件名/正文里的特殊字符会导致打印失败。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)
    settings = get_settings()

    print(f"Milvus collection: {settings.milvus_collection}")
    print(f"混合检索: {settings.hybrid_retrieval_enabled}")

    # 触发 schema 校验：collection 不支持混合检索时会直接报错退出。
    capabilities = vector_store.collection_capabilities()
    print(
        "collection 能力："
        f"sparse_field={capabilities['has_sparse_field']} "
        f"bm25_function={capabilities['has_bm25_function']} "
        f"text_analyzer={capabilities['text_analyzer_enabled']}"
    )

    with session_scope() as db:
        stmt = select(KnowledgeFile)
        if args.file_id:
            stmt = stmt.where(KnowledgeFile.id.in_(args.file_id))
        elif not args.include_failed:
            stmt = stmt.where(KnowledgeFile.status == FileStatus.READY)
        rows = db.scalars(stmt.order_by(KnowledgeFile.id)).all()
        snapshots = [
            (row.id, row.file_name, row.storage_path, row.status) for row in rows
        ]

    if not snapshots:
        print("没有需要处理的文件。")
        return 0

    print(f"共 {len(snapshots)} 个文件待处理。")
    failed = 0

    for file_id, file_name, storage_path, status in snapshots:
        path = _resolve_storage_path(storage_path)
        print(f"[{file_id}] {file_name} status={status} path={path}")

        if not path.exists():
            print("  跳过：源文件不存在")
            failed += 1
            continue

        if args.dry_run:
            continue

        try:
            # add() 生成随机 node_id，不清旧向量会导致重复。
            vector_store.delete_by_file(file_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过：清理旧向量失败：{exc}")
            failed += 1
            continue

        ingest_service.process_file(file_id)

        with session_scope() as db:
            row = db.get(KnowledgeFile, file_id)
            if row is None:
                print("  结果：文件记录已不存在")
                failed += 1
                continue
            print(f"  结果：{row.status} chunks={row.chunk_count}")
            if row.status != FileStatus.READY:
                failed += 1
                if row.error_message:
                    print(f"  错误：{row.error_message[:300]}")

    print(f"完成：失败 {failed} 个。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
