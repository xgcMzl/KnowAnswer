"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from know_answer.app.routes import chat, health, knowledge
from know_answer.core.config import get_settings
from know_answer.core.db import init_db
from know_answer.core.errors import register_exception_handlers
from know_answer.rag.embedding import get_embed_model
from know_answer.rag.vector_store import get_vector_store
from know_answer.services import ingest_service
from know_answer.services.ingest_service import recover_stale_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _warmup() -> None:
    """后台预热模型与向量库，避免首请求长时间等待。"""
    try:
        get_embed_model()
        logger.info("Embedding 模型预热完成")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding 模型预热失败：%s", exc)
    try:
        get_vector_store()
        logger.info("Milvus collection 就绪")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Milvus 初始化失败：%s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    recover_stale_files()
    threading.Thread(target=_warmup, name="knowanswer-warmup", daemon=True).start()
    logger.info("KnowAnswer 后端启动完成")
    try:
        yield
    finally:
        ingest_service.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(knowledge.router)
    app.include_router(chat.router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": settings.app_name, "docs": "/docs", "api": "/api"}

    return app


app = create_app()
