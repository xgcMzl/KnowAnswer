# KnowAnswer

基于 **FastAPI + LangChain + LlamaIndex + Milvus** 的知识库问答（RAG）系统。

- 后端：FastAPI + LangChain（DeepSeek 流式）+ LlamaIndex（解析/切分/Embedding/检索）
- 前端：Vue 3 + TypeScript + Element Plus + Pinia（`frontend/`）
- 文档：PDF / DOCX / Markdown / TXT
- 切分：LlamaIndex `SentenceSplitter`，父子两级切片（父块存 SQLite，子块进 Milvus）
- Embedding：本地 `BAAI/bge-small-zh-v1.5`（512 维，CPU）
- 向量库：Milvus（collection `knowanswer_chunks_v2`，按知识库/文件过滤）
- 检索：稠密向量 + Milvus 内置 BM25 双路召回，服务端 RRF 融合
- 元数据：SQLite + SQLAlchemy

> 不含登录鉴权，默认只应监听 `127.0.0.1`。

## 快速开始

### 1. 后端

```bash
uv sync
cp .env.example .env        # 填入 DEEPSEEK_API_KEY
uv run uvicorn know_answer.app._app:app --reload --port 8000
```

接口文档：<http://127.0.0.1:8000/docs>

### 2. 前端

```bash
cd frontend
npm install
npm run dev
```

访问 <http://localhost:5173>。开发模式下 Vite 会把 `/api` 代理到 `http://127.0.0.1:8000`。

生产构建：

```bash
npm run build
```

构建产物在 `frontend/dist/`。后端启动时会自动托管该目录，因此构建完成后直接访问 <http://127.0.0.1:8000/chat> 即可（`/`、`/chat`、`/knowledge` 都由 SPA 处理）。

也可以由 Nginx 等静态服务托管，并把 `/api` 反向代理到后端；如前后端不同源，构建时设置 `VITE_API_BASE=http://后端地址`，同时把该前端地址加入后端 `CORS_ORIGINS`。

## 前端功能

- 对话页：会话列表、新建/重命名/删除会话、知识库范围选择、Markdown 回答、引用来源展示、流式生成与停止
- 知识库页：知识库增删改、拖拽上传、文件状态轮询（排队中/入库中/可检索/失败）、切片数、删除文件
- 侧边栏：服务健康状态（数据库 / Milvus / Embedding / LLM）

## 环境变量

见 `.env.example`。关键项：

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 必填 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | Milvus 地址与 collection（混合检索需用 `knowanswer_chunks_v2`） |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM` | 默认 bge-small-zh-v1.5 / 512 |
| `PARENT_CHILD_ENABLED` | 父子切片开关，默认 `true` |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | 父块大小/重叠（token），默认 1024 / 128 |
| `HYBRID_RETRIEVAL_ENABLED` | 混合检索开关，默认 `true` |
| `RAG_RRF_K` / `RAG_SPARSE_ANALYZER` | RRF 参数（默认 60）与 BM25 分词器（默认 `chinese`） |
| `RAG_TOP_K` | 最终来源条数，默认 6 |
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

### 父子切片与混合检索

- 入库：先按 `PARENT_CHUNK_SIZE` 切父块（只存 SQLite 表 `parent_chunks`），再把每个父块按 `CHUNK_SIZE` 切子块写入 Milvus；子块 metadata 带 `parent_id`。
- 检索：稠密向量 + BM25 双路召回，Milvus `RRFRanker` 融合，然后按 `parent_id` 折叠并回表取父块原文作为 prompt 上下文。
- 混合检索要求 collection 在创建时就带 BM25 内置函数。旧的 `knowanswer_chunks` 不满足该 schema，必须使用新 collection（默认 `knowanswer_chunks_v2`）并重新入库。

旧数据迁移（迁移期间请停止上传与其他入库任务）：

```bash
uv run python -m know_answer.tools.reingest_parent_child --dry-run
uv run python -m know_answer.tools.reingest_parent_child --include-failed
uv run python -m know_answer.tools.smoke_search --query "某个精确关键词"
```

`smoke_search` 会分别打印 dense / sparse / hybrid 的召回结果；hybrid 分数是 RRF 融合分（0.0x 量级），不是余弦相似度。

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
