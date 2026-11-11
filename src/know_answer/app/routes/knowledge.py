"""知识库与文件接口。"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from sqlalchemy.orm import Session

from know_answer.core.db import get_db
from know_answer.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseList,
    KnowledgeBaseOut,
    KnowledgeBaseUpdate,
    KnowledgeFileList,
    KnowledgeFileOut,
)
from know_answer.services import ingest_service, knowledge_service

router = APIRouter(prefix="/api", tags=["knowledge"])


@router.post(
    "/knowledge-bases", response_model=KnowledgeBaseOut, status_code=201
)
def create_knowledge_base(
    payload: KnowledgeBaseCreate, db: Session = Depends(get_db)
) -> KnowledgeBaseOut:
    return knowledge_service.create_knowledge_base(
        db, payload.name, payload.description
    )


@router.get("/knowledge-bases", response_model=KnowledgeBaseList)
def list_knowledge_bases(db: Session = Depends(get_db)) -> KnowledgeBaseList:
    return KnowledgeBaseList(items=knowledge_service.list_knowledge_bases(db))


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseOut)
def get_knowledge_base(kb_id: int, db: Session = Depends(get_db)) -> KnowledgeBaseOut:
    return knowledge_service.get_knowledge_base(db, kb_id)


@router.patch("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseOut)
def update_knowledge_base(
    kb_id: int, payload: KnowledgeBaseUpdate, db: Session = Depends(get_db)
) -> KnowledgeBaseOut:
    return knowledge_service.update_knowledge_base(
        db, kb_id, name=payload.name, description=payload.description
    )


@router.delete("/knowledge-bases/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: int, db: Session = Depends(get_db)) -> None:
    knowledge_service.delete_knowledge_base(db, kb_id)


@router.post(
    "/knowledge-bases/{kb_id}/files",
    response_model=KnowledgeFileOut,
    status_code=202,
)
async def upload_file(
    kb_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeFileOut:
    row = await knowledge_service.save_upload(db, kb_id, file)
    background_tasks.add_task(ingest_service.submit, row.id)
    return KnowledgeFileOut.model_validate(row)


@router.get(
    "/knowledge-bases/{kb_id}/files", response_model=KnowledgeFileList
)
def list_files(kb_id: int, db: Session = Depends(get_db)) -> KnowledgeFileList:
    rows = knowledge_service.list_files(db, kb_id)
    return KnowledgeFileList(
        items=[KnowledgeFileOut.model_validate(row) for row in rows]
    )


@router.get("/files/{file_id}", response_model=KnowledgeFileOut)
def get_file(file_id: int, db: Session = Depends(get_db)) -> KnowledgeFileOut:
    return KnowledgeFileOut.model_validate(
        knowledge_service.get_file_or_404(db, file_id)
    )


@router.delete("/files/{file_id}", status_code=204)
def delete_file(file_id: int, db: Session = Depends(get_db)) -> None:
    knowledge_service.delete_file(db, file_id)
