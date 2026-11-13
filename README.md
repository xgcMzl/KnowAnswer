# KnowAnswer

基于 **FastAPI + LangChain + LlamaIndex + Milvus** 的知识库问答（RAG）系统。

- 后端：FastAPI + LangChain（DeepSeek 流式）+ LlamaIndex（解析/切分/Embedding/检索）
- 前端：Vue 3 + TypeScript + Element Plus + Pinia（`frontend/`）
- 文档：PDF / DOCX / Markdown / TXT
- 切分：LlamaIndex `SentenceSplitter`，父子两级切片（父块存 SQLite，子块进 Milvus）
- Embedding：本地 `BAAI/bge-small-zh-v1.5`（512 维，CPU）
- 向量库：Milvus（collection `knowanswer_chunks_v2`，按知识库/文件过滤）
- 检索：稠密向量 + Milvus 内置 BM25 双路召回，服务端 RRF 融合；按父块折叠后取 ≤6 条，整轮上下文 ≤8000 字符
- 意图识别：LangChain Tool 轻量路由（`bind_tools`），模型在「检索知识库」与「直接回答」间二选一，常规问题跳过 RAG
- 元数据：SQLite + SQLAlchemy

> 不含登录鉴权，默认只应监听 `127.0.0.1`。

架构与实现细节见 [`doc/knowanswer_architecture.md`](doc/knowanswer_architecture.md)。

## 环境要求

- Python ≥ 3.12 + [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18（Vite 6）
- 可访问的 Milvus（`MILVUS_URI` 可指向自建实例或 Zilliz Cloud）
- 首次启动会下载 Embedding 模型（默认 `BAAI/bge-small-zh-v1.5`）

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
- 意图提示：流式气泡显示「正在结合知识库回答…」/「直接回答（未使用知识库）…」；常规回答的历史消息带「未使用知识库」标签
- 知识库页：知识库增删改、拖拽上传、文件状态轮询（排队中/入库中/可检索/失败）、切片数、删除文件
- 侧边栏：服务健康状态（数据库 / Milvus / Embedding / LLM）

## 问答流程

```
POST /api/chat/stream
  ├─ 1. 校验会话、读取最近历史、保存 user 消息
  ├─ 2. 意图识别（LangChain Tool 路由，一次性非流式调用）
  │      search_knowledge_base → rag      answer_directly → general
  ├─ 3a. rag：召回 18 个子块（RAG_TOP_K × RAG_CHILD_TOP_K_MULTIPLIER）
  │         → 服务端 RRF 融合 → 按 parent_id 折叠 → 取 ≤6 个父块
  │         → 单块 ≤1600 字、整轮 ≤8000 字 → 拼进 system prompt
  └─ 3b. general：跳过检索，使用通用 system prompt
          ↓
      流式生成 → 保存 assistant 消息
```

- `rag` 分支下，模型会把追问改写成独立问句（补全「它 / 这个 / 那第二条」等指代）；检索用改写后的 query，回答仍针对用户原问题。
- 路由失败不会中断问答：超时、异常、无工具调用等按 `INTENT_ROUTER_FALLBACK_INTENT`（默认 `rag`）兜底。
- 系统**没有精排（cross-encoder rerank）**环节，排序仅由 Milvus 服务端 RRF 融合完成。详见架构文档。

## 环境变量

见 `.env.example`。关键项：

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 必填 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | Milvus 地址与 collection（混合检索需用 `knowanswer_chunks_v2`） |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM` | 默认 bge-small-zh-v1.5 / 512 |
| `PARENT_CHILD_ENABLED` | 父子切片开关，默认 `true` |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | 父块大小/重叠（token），默认 1024 / 128 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 子块大小/重叠（token），默认 512 / 64 |
| `HYBRID_RETRIEVAL_ENABLED` | 混合检索开关，默认 `true` |
| `RAG_RRF_K` / `RAG_SPARSE_ANALYZER` | RRF 参数（默认 60）与 BM25 分词器（默认 `chinese`） |
| `RAG_TOP_K` | 最终父块条数上限，默认 6 |
| `RAG_CHILD_TOP_K_MULTIPLIER` | 子块召回放大系数，默认 3（`6 × 3 = 18` 个子块） |
| `RAG_PARENT_MAX_CHARS` / `RAG_CONTEXT_MAX_CHARS` | 单块 / 整轮上下文最大字符数，默认 1600 / 8000 |
| `INTENT_ROUTING_ENABLED` | 意图识别开关，默认 `true`；关闭后所有问题都走知识库检索 |
| `INTENT_ROUTER_TOOL_CHOICE` | 工具选择策略，默认 `auto`；端点支持时可改 `required` |
| `INTENT_ROUTER_TIMEOUT_SECONDS` | 意图识别超时，默认 5 秒 |
| `INTENT_ROUTER_FALLBACK_INTENT` | 路由失败时的兜底意图，默认 `rag` |
| `MEMORY_MAX_MESSAGES` / `MEMORY_MAX_CHARS` | 对话记忆条数 / 字符上限，默认 10 / 4000 |
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
data: {"type":"intent","intent":"rag"}            # rag | general
data: {"type":"sources","sources":[...]}          # general 时为 []
data: {"type":"delta","content":"..."}
data: {"type":"done","conversation_id":1,"message_id":2}
data: [DONE]
```

- `intent` 帧在前端用于显示「是否使用了知识库」。
- `general` 分支不检索，`sources` 为 `[]`；落库时 `sources` 记为 `null`，用于和「检索了但没命中」（`[]`）区分。
- 错误帧：`{"type":"error","code":"...","message":"..."}`。检索阶段的错误仍返回标准 HTTP JSON（如 503），只有进入流式后的错误才用 SSE 错误帧。

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

当前 48 个用例，覆盖父子切分、检索折叠与预算、混合检索降级、Prompt 组装、意图路由（含超时/异常/未知工具等分支）与流式分流。全部不依赖真实 Milvus / DeepSeek；端到端验证需要 `.env` 中的真实配置。

## 目录

```
src/know_answer/
├── app/          FastAPI 应用、路由、SPA 托管
├── core/         配置、数据库、异常
├── db/           SQLAlchemy 模型、父块读写
├── schemas/      Pydantic 请求/响应
├── services/     业务编排（知识库、入库、会话）
├── rag/          Embedding、Milvus、意图路由、检索、Prompt、LLM
├── data_index/   文档解析/切分/入库实现
├── models/       早期 LangChain 封装示例（主链路未使用）
└── tools/        运维 CLI（smoke_search、reingest_parent_child）
frontend/         Vue 3 + TypeScript 前端
tests/            pytest 单元测试
doc/              架构文档
```

## 已知限制

- 不含登录鉴权与多租户，默认只应监听 `127.0.0.1`。
- 会话绑定单个知识库；`knowledge_base_id = null` 表示「全部知识库」（检索时不过滤），不是「没有知识库」。
- 无精排（rerank）环节，排序仅由 RRF 融合完成；`RAG_TOP_K` 是对排序精度缺失的补偿。
- 无数据库迁移工具（未引入 Alembic），表结构变更需手工处理。
- 入库任务在进程内单线程队列执行，进程重启后中断的任务会标记为 `failed`，需重新上传。
- 意图只在 SSE 中返回、不落库；历史消息通过 `sources === null` 判断「未使用知识库」。
- 组件均为进程内单例，建议单进程部署。
