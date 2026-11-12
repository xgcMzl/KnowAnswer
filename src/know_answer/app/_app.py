"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

    if not _mount_frontend(app, settings.app_name):
        @app.get("/")
        async def root() -> dict[str, str]:
            return {"name": settings.app_name, "docs": "/docs", "api": "/api"}

    return app


def _mount_frontend(app: FastAPI, app_name: str) -> bool:
    """存在 frontend/dist 时，由后端直接托管 SPA（单进程部署）。"""
    project_root = Path(__file__).resolve().parents[3]
    dist = (project_root / "frontend" / "dist").resolve()
    if not dist.is_dir():
        return False

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "NOT_FOUND", "message": f"接口不存在：/{full_path}"}},
            )
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")

    logger.info("%s 将托管前端静态文件：%s", app_name, dist)
    return True


app = create_app()
