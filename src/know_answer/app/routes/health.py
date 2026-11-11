"""健康检查。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from know_answer.core.db import session_scope
from know_answer.rag import vector_store
from know_answer.rag.embedding import is_embedding_loaded
from know_answer.rag.llm import is_llm_configured

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict:
    database = "ok"
    try:
        with session_scope() as db:
            db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        database = "error"

    try:
        milvus = await asyncio.wait_for(
            run_in_threadpool(vector_store.health), timeout=5.0
        )
    except (asyncio.TimeoutError, TimeoutError):
        milvus = {"status": "timeout"}

    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "milvus": milvus,
        "embedding_model": "loaded" if is_embedding_loaded() else "not_loaded",
        "llm": "configured" if is_llm_configured() else "missing",
    }
