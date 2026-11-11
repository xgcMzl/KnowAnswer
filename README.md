# KnowAnswer

基于 **FastAPI + LangChain + LlamaIndex + Milvus** 的知识库问答（RAG）后端。

- 文档：PDF / DOCX / Markdown / TXT
- 切分：LlamaIndex `SentenceSplitter`
- Embedding：本地 `BAAI/bge-small-zh-v1.5`（512 维，CPU）
- 向量库：Milvus（collection `knowanswer_chunks`，按知识库/文件过滤）
- 生成：DeepSeek（LangChain，SSE 流式）
- 元数据：SQLite + SQLAlchemy

> v1 只包含后端 API，不含登录鉴权。默认只应监听 `127.0.0.1`。

## 快速开始

```bash
uv sync
cp .env.example .env        # 填入 DEEPSEEK_API_KEY
uv run uvicorn know_answer.app._app:app --reload --port 8000
```

文档：<http://127.0.0.1:8000/docs>

## 环境变量

见 `.env.example`。关键项：

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 必填 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | Milvus 地址与 collection |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM` | 默认 bge-small-zh-v1.5 / 512 |
| `RAG_TOP_K` | 召回条数，默认 6 |
| `DATABASE_URL` | 默认 `sqlite:///./data/knowanswer.db` |

## API 概览

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查（DB / Milvus / 模型） |
| POST/GET | `/api/knowledge-bases` | 创建 / 列出知识库 |
| GET/PATCH/DELETE | `/api/knowledge-bases/{id}` | 知识库详情 / 更新 / 删除 |
| POST | `/api/knowledge-bases/{id}/files` | 上传文件（202，后台入库） |
| GET | `/api/knowledge-bases/{id}/files` | 文件列表 |
| GET/DELETE | `/api/files/{id}` | 文件状态 / 删除 |
| POST/GET | `/api/conversations` | 创建 / 列出会话 |
| PATCH/DELETE | `/api/conversations/{id}` | 重命名 / 删除会话 |
| GET | `/api/conversations/{id}/messages` | 消息列表 |
| POST | `/api/chat/stream` | SSE 流式问答 |

### 文件状态

`pending → processing → ready | failed`。上传后轮询 `GET /api/files/{id}` 直到 `ready`。

### SSE 帧

```
data: {"type":"sources","sources":[...]}
data: {"type":"delta","content":"..."}
data: {"type":"done","conversation_id":1,"message_id":2}
data: [DONE]
```

错误帧：`{"type":"error","code":"...","message":"..."}`。

## 示例

```bash
# 创建知识库
curl -X POST http://127.0.0.1:8000/api/knowledge-bases \
  -H "Content-Type: application/json" -d '{"name":"default"}'

# 上传
curl -X POST http://127.0.0.1:8000/api/knowledge-bases/1/files \
  -F "file=@data/example.txt"

# 等待 ready 后创建会话
curl -X POST http://127.0.0.1:8000/api/conversations \
  -H "Content-Type: application/json" -d '{"knowledge_base_id":1}'

# 流式问答
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":1,"question":"这份文档讲了什么？"}'
```

> Windows Git Bash 下直接传中文 JSON 可能出现编码问题，建议把请求体写入 UTF-8 文件后用 `--data-binary @file.json`。

## 测试

```bash
uv run pytest
```

单元测试不依赖真实 Milvus / DeepSeek；端到端验证需要 `.env` 中的真实配置。

## 目录

```
src/know_answer/
├── app/          FastAPI 应用、路由、依赖
├── core/         配置、数据库、异常
├── db/           SQLAlchemy 模型
├── schemas/      Pydantic 请求/响应
├── services/     业务编排（知识库、入库、会话）
├── rag/          Embedding、Milvus、检索、Prompt、LLM
└── data_index/   既有文档解析/切分/入库实现
```
