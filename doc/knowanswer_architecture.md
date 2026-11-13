# KnowAnswer 架构与实现详解

> 本文档描述当前 `master` 分支的代码实现（含「用户意图识别」改造）。
> 文中路径均相对项目根目录，行号以当前代码为准，便于跳转阅读。

---

## 1. 项目概览

KnowAnswer 是一个**私有知识库问答（RAG）系统**：用户上传 PDF / DOCX / Markdown / TXT 文档，系统解析、切分、向量化后存入 Milvus；提问时先做意图识别，需要时检索知识库，再由大模型流式作答并给出引用来源。

### 1.1 技术栈

| 层次 | 技术 |
| --- | --- |
| Web 框架 | FastAPI（Python ≥ 3.12）+ Uvicorn |
| 依赖管理 | `uv`（`pyproject.toml` + `uv.lock`） |
| 业务元数据 | SQLite + SQLAlchemy 2.0（可直接迁移到 MySQL） |
| 文档解析/切分 | LlamaIndex（`llama-index-readers-file`、`SentenceSplitter`） |
| Embedding | LlamaIndex `HuggingFaceEmbedding`，默认 `BAAI/bge-small-zh-v1.5`（512 维，CPU） |
| 向量库 | Milvus（`llama-index-vector-stores-milvus`），默认 collection `knowanswer_chunks_v2` |
| 对话模型 | DeepSeek，经 LangChain 1.4 `init_chat_model` 初始化 |
| 意图识别 | LangChain Tool Calling（`@tool` + `bind_tools`） |
| 前端 | Vue 3 + TypeScript + Vite 6 + Pinia + Element Plus |
| 测试 | pytest（同步用例；异步用 `asyncio.run` 包装） |

### 1.2 目录结构

```
KnowAnswer/
├── src/know_answer/
│   ├── app/                    FastAPI 应用层
│   │   ├── _app.py             应用创建、lifespan、SPA 托管
│   │   ├── deps.py             依赖再导出
│   │   └── routes/             chat.py / knowledge.py / health.py
│   ├── core/                   基础设施
│   │   ├── config.py           pydantic-settings 配置
│   │   ├── db.py               引擎、Session、建表
│   │   └── errors.py           业务异常体系 + 统一错误响应
│   ├── db/                     数据访问
│   │   ├── models.py           SQLAlchemy 模型
│   │   └── parent_store.py     父块读写
│   ├── schemas/                Pydantic 请求/响应模型
│   ├── services/               业务编排（无 HTTP 依赖）
│   │   ├── chat_service.py     会话/消息/流式前后置
│   │   ├── knowledge_service.py 知识库与文件
│   │   └── ingest_service.py   后台入库任务
│   ├── rag/                    RAG 主链路
│   │   ├── router.py           意图识别（Tool 路由）
│   │   ├── retrieval.py        检索编排 + 父子块回填
│   │   ├── vector_store.py     Milvus 适配
│   │   ├── embedding.py        Embedding 单例
│   │   ├── prompt.py           Prompt 组装
│   │   ├── llm.py              DeepSeek 异步流式封装
│   │   └── ingestion.py        入库适配层
│   ├── data_index/
│   │   └── doc_loading.py      解析/切分/写 Milvus 的实现
│   ├── models/                 早期 LangChain 封装示例（主链路未使用）
│   └── tools/                  运维 CLI 脚本
├── frontend/                   Vue 3 前端
├── tests/                      48 个单元测试
├── data/                       SQLite 库与上传文件
└── doc/                        文档
```

---

## 2. 总体架构

### 2.1 分层视图

```
┌───────────────────────────────────────────────────────────────┐
│  前端（Vue 3 + TS）                                            │
│  ChatView / KnowledgeView / Pinia store / fetch-SSE 解析器      │
└───────────────┬───────────────────────────────────────────────┘
                │ HTTP / SSE
┌───────────────▼───────────────────────────────────────────────┐
│  接口层  app/routes/                                           │
│  chat.py     knowledge.py     health.py                       │
│  · 参数校验（Pydantic）  · SSE 帧构造  · 依赖注入 Session       │
└───────────────┬───────────────────────────────────────────────┘
                │ 显式参数调用（多数 service 接收 db: Session）
┌───────────────▼───────────────────────────────────────────────┐
│  业务编排层  services/                                         │
│  chat_service      knowledge_service      ingest_service       │
│  · 事务边界        · 文件校验/落盘         · 单线程后台入库队列 │
└───┬───────────────────┬──────────────────────────┬────────────┘
    │                   │                          │
┌───▼─────────────┐ ┌───▼──────────────┐ ┌─────────▼───────────┐
│  rag/ 问答链路   │ │ db/ 元数据       │ │ rag/ingestion +      │
│  router         │ │ models           │ │ data_index/ 解析入库  │
│  retrieval      │ │ parent_store     │ │ doc_loading          │
│  prompt / llm   │ └──────┬───────────┘ └─────────┬───────────┘
└───┬─────────────┘        │                       │
    │                      │                       │
┌───▼──────────────────────▼───────────────────────▼───────────┐
│  外部依赖                                                     │
│  Milvus（子块向量 + BM25 稀疏向量）                            │
│  SQLite（业务元数据 + 父块正文）                               │
│  HuggingFace 本地 Embedding 模型                              │
│  DeepSeek Chat API                                            │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 两条核心链路

| 链路 | 入口 | 落点 |
| --- | --- | --- |
| **写入**：文档入库 | `POST /api/knowledge-bases/{id}/files` | Milvus（子块）+ SQLite（父块、文件状态） |
| **读取**：问答 | `POST /api/chat/stream` | SSE 流式回答 + 引用来源 |

### 2.3 进程内单例与线程模型

所有重资源组件都是**模块级单例 + 双检锁**，首次使用时初始化：

| 组件 | 位置 | 初始化开销 |
| --- | --- | --- |
| Embedding 模型 | `rag/embedding.py::get_embed_model` | 高（本地模型加载） |
| Milvus 适配器 | `rag/vector_store.py::get_vector_store` | 中（建连 + schema 校验） |
| 对话模型 | `rag/llm.py::get_chat_model` | 低（HTTP 客户端） |
| Ingestor | `rag/ingestion.py::get_ingestor` | 低 |

每个模块都提供 `reset_*()` 供测试清空单例。

启动时 `app/_app.py:31-42` 用**独立 daemon 线程**做预热（Embedding + Milvus），避免首个请求长时间阻塞：

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    recover_stale_files()          # 把上次中断的入库任务标记为 failed
    threading.Thread(target=_warmup, daemon=True).start()
    yield
    ingest_service.shutdown()
```

**线程池边界**是明确的：

- 同步阻塞操作（检索、SQLite 读写）→ `run_in_threadpool`；
- 异步网络调用（LLM 路由、LLM 流式）→ 直接 `await`；
- 文档入库 → 独立的单线程 `ThreadPoolExecutor`。

---

## 3. 数据模型

### 3.1 SQLite（`db/models.py`）

```
knowledge_bases (知识库)
   │ 1
   │ n
knowledge_files (文件)          status: pending / processing / ready / failed
   │ 1
   │ n
parent_chunks (父块正文)        parent_id = "{file_id}:{parent_index}"
   └── 主键 parent_id，唯一约束 (file_id, parent_index)

conversations (会话)
   │ 1
   │ n
messages (消息)                 role: user / assistant；sources: JSON
```

关键设计：

- **`parent_chunks` 只存 SQLite，不进 Milvus。** 父块动辄 1000+ token，全量写向量库会显著推高存储与检索成本；而它只用于「命中子块后回填上下文」，按主键批量查 SQLite 足够快。
- `messages.sources` 是 JSON 列：`null` = 本轮未使用知识库；`[]` = 用了知识库但没命中；非空 = 引用来源列表。
- `UTCDateTime` 类型装饰器：SQLite 不保存时区，统一以 UTC 裸时间落库、读取时补回 `timezone.utc`。
- 无 Alembic，建表只靠 `Base.metadata.create_all`。**新增列不会自动迁移**，因此「意图」这种展示信息刻意没有落库（见 §5.4）。

### 3.2 Milvus collection

默认 `knowanswer_chunks_v2`，由 `rag/vector_store.py::_build_vector_store` 创建：

| 字段 | 说明 |
| --- | --- |
| `embedding` | 稠密向量（维度 = `EMBEDDING_DIM`，默认 512，COSINE） |
| `text` | 子块原文（BM25 的输入字段） |
| `sparse_embedding` | BM25 内置函数生成的稀疏向量 |
| `knowledge_base_id` / `file_id` / `file_name` | 业务标量字段，用于过滤与删除 |

混合检索配置（`vector_store.py:64-78`）：

```python
enable_sparse=True,
sparse_embedding_function=BM25BuiltInFunction(
    input_field_names="text",
    output_field_names="sparse_embedding",
    analyzer_params={"type": settings.rag_sparse_analyzer},   # 默认 chinese（jieba）
),
hybrid_ranker="RRFRanker",
hybrid_ranker_params={"k": settings.rag_rrf_k},               # 默认 60
```

> **为什么必须显式传 `BM25BuiltInFunction`**：复用旧 collection 时，适配器默认会回退到 `BGEM3SparseEmbeddingFunction`，依赖 FlagEmbedding + BGE-M3，本项目并未安装。
>
> **为什么新开 `_v2` collection**：Milvus 的 BM25 函数必须在建 collection 时声明，旧 collection 无法补加。

---

## 4. 写入链路：文档入库

### 4.1 时序

```
POST /api/knowledge-bases/{id}/files
  └─ knowledge_service.save_upload           ① 校验 + 流式落盘 + 建 pending 记录
       └─ BackgroundTasks.add_task(ingest_service.submit, file_id)
            └─ ThreadPoolExecutor(max_workers=1).submit(process_file)
                 ├─ 事务1：status = processing
                 ├─ ingest_document                   ② 解析 → 切分 → embedding → 写 Milvus
                 └─ 事务2：父块落库 + status = ready   ③ 同一事务，避免半成品状态
```

接口立即返回 **202**，前端轮询 `GET /api/files/{id}` 获取状态。

### 4.2 上传校验（`services/knowledge_service.py:162-225`）

- 扩展名白名单：`.pdf` `.docx` `.md` `.txt`（大小写不敏感，取 `Path(filename).name` 防路径穿越）；
- 1MB 分块流式写盘，超过 `MAX_UPLOAD_MB`（默认 20）立即中止并删除临时文件；
- 文件名为 `uuid4().hex + 后缀`，原始名只存数据库；
- 磁盘写入或建表失败都会回滚并清理残留文件。

### 4.3 解析与切分（`data_index/doc_loading.py`）

**解析**（`_build_reader`）：PDF 用 `PDFReader(return_full_document=False)`，**每页生成一个 Document** 以保留 `page_label`；DOCX 用 `DocxReader`；TXT 用 `FlatReader`（编码失败时按 UTF-8 容错解码）；MD 用 `MarkdownReader`。

每个节点的 metadata 至少包含：`file_name` `file_type` `source_path` + 业务注入的 `knowledge_base_id` `file_id` `file_name`。

**父子两级切分**（`_split`，第 678-751 行）：

```
Document（PDF 的一页）
   └─ parent_splitter(1024 / 128)  →  父块
         └─ child_splitter(512 / 64)  →  子块（进 Milvus）
```

规则：

- **父块不跨 Document**，避免 PDF 的 `page_label` 串页；
- `parent_index` / `chunk_index` 按整份文件**全局递增**；
- `parent_id = f"{file_key}:{parent_index}"`，业务链路里 `file_key = file_id`；
- 子块 metadata 额外写入 `parent_id`、`parent_index`、`child_index_in_parent`、`chunk_index`；
- `PARENT_CHILD_ENABLED=false` 时退化为扁平切分，子块没有 `parent_id`，检索侧走旧逻辑。

> **为什么不直接用 `get_nodes_from_documents`**：LlamaIndex 的该 API 会把 metadata 计入 token 预算，chunk_size 较小时会抛 `Metadata length is longer than chunk size`。这里对正文单独调用 `split_text`，节点 metadata 自行复制。

### 4.4 向量化与写入

- `_embed_nodes` 只对 `MetadataMode.NONE`（纯正文）生成向量，逐节点写入 `node.embedding`；
- 校验：向量数量、非空、维度一致性，任一不符直接抛错；
- `vector_store.add(nodes, force_flush=True)` 让数据尽快对检索可见；
- 返回值 `IngestResult` 携带 `node_count`（子块数）、`parent_count`、`parent_chunks` 草稿。

### 4.5 状态机与失败处理

```
pending ──process_file开始──> processing ──成功──> ready
   │                               │
   └──────服务重启──────────────────┴──异常──────> failed (+ error_message)
```

`process_file`（`services/ingest_service.py:29-83`）的异常分支会**三段式清理**，保证不留脏数据：

1. `vector_store.delete_by_file(file_id)` —— 清残留子块；
2. `parent_store.delete_by_file(...)` —— 清残留父块；
3. `status = failed`，`error_message` 截断到 1000 字符。

`recover_stale_files()` 在启动时把所有 `pending/processing` 标记为 `failed`，因为进程重启后线程池里的任务已丢失。

> **为什么是 `max_workers=1`**：本地 Embedding 在 CPU 上跑，并发入库只会互相抢 CPU 并让所有任务都变慢；串行反而总吞吐更稳定，也让状态机更简单。

---

## 5. 读取链路：问答

### 5.1 时序总览

`app/routes/chat.py::chat_stream`

```
POST /api/chat/stream
  │
  ├─ A. run_in_threadpool(chat_service.prepare_stream)       短事务：
  │     校验会话 → 解析 knowledge_base_id → 读最近 N 条历史 → 写 user 消息
  │
  ├─ B. await classify_intent(...)                           意图识别（非流式）
  │     └─ bind_tools → 模型选工具 → IntentDecision(intent, query)
  │
  ├─ C. intent == "rag"
  │       run_in_threadpool(retrieve, 改写query, kb_id)      检索（失败走 JSON 503）
  │       build_messages(原问题, sources, history)
  │     else
  │       build_general_messages(原问题, history)             不检索
  │
  └─ D. StreamingResponse(event_stream())
        intent 帧 → sources 帧 → delta 帧* → done 帧 → [DONE]
                              └─ 断开检测 / 空回答检测 / 异常帧
                              └─ 结束后 run_in_threadpool(save_assistant)
```

**为什么路由和检索都在 `StreamingResponse` 之前完成**：这样检索失败仍是标准 FastAPI JSON 错误（503），前端 `streamChat` 的 `!response.ok` 分支就能处理；一旦进入流式，就只能用 SSE error 帧，错误处理路径会分叉。

### 5.2 意图识别（`rag/router.py`）

这是本系统唯一的「模型决策」环节，实现方式是**单次 Tool Calling 分类**，不是 Agent 循环。

#### 5.2.1 两个工具

```python
@tool(parse_docstring=True)
def search_knowledge_base(query: str) -> str: ...   # 需要查文档
@tool(parse_docstring=True)
def answer_directly(question: str) -> str: ...      # 直接回答

INTENT_TOOLS = [search_knowledge_base, answer_directly]
```

要点：

- `parse_docstring=True` 把中文 docstring 正文转成 tool description，`Args:` 段转成 JSON Schema；
- docstring 同时写「适用场景」和「**不要用于**」——负例比正例更能压制误选；
- **函数体是信号桩**，返回固定字符串。真正的检索由 `retrieve()` 执行，这两个工具只是给模型看的两个选项，不要交给 Agent 执行。

#### 5.2.2 调用与解析

```python
bound = get_chat_model().bind_tools(INTENT_TOOLS, tool_choice=tool_choice)
async with asyncio.timeout(settings.intent_router_timeout_seconds):
    response = await bound.ainvoke(messages)        # system + 截断历史 + 原问题
```

- `tool_choice` 默认 `auto`，可配 `required`；
- 温度复用全局 `LLM_TEMPERATURE=0.0`，结果更确定；
- 历史只取最近 `INTENT_ROUTER_HISTORY_MESSAGES`（4）条、且不超过 `INTENT_ROUTER_HISTORY_CHARS`（2000）字，控制首字延迟。

解析规则（`_parse_decision`）：

1. 优先读 `AIMessage.tool_calls`（LangChain 已归一化）；
2. 为空时回退 `additional_kwargs["tool_calls"]`，并兼容 `arguments` 是 JSON 字符串的代理端点；
3. 多个调用取第一个可识别的，未知工具名跳过；
4. `rag` 分支的 `query` 会被截断到 `INTENT_ROUTER_QUERY_MAX_CHARS`，为空则由编排层回退到用户原问题。

#### 5.2.3 Prompt 的关键约束

`ROUTER_SYSTEM_PROMPT` 的三条核心规则：

1. **只看当前问题本身**判断是否需要上传文档里的信息；
2. **对话历史只用于补全指代，不能决定意图**——上一轮检索过知识库不代表本轮也要检索；
3. 拿不准时：提到具体内部/产品/项目/政策信息才选 `search_knowledge_base`，否则 `answer_directly`。

> 实测教训：初版规则写「拿不准但可能涉及私有信息时优先 rag」，在上文聊过知识库的情况下会把「讲个笑话」误判成 rag。改成第 2、3 条后，6 个典型问题全部判对。

#### 5.2.4 兜底

`classify_intent` **除 `CancelledError` 外不抛异常**：

| 情况 | 结果 |
| --- | --- |
| 开关关闭 / 未配置 API Key | `rag`，且不发起模型调用 |
| 超时（默认 5s） | `rag`（`decided_by="fallback"`） |
| 网络/API 异常、非 `AIMessage` | `rag` |
| 无 `tool_calls` / 全部未知工具 | `rag` |

兜底意图由 `INTENT_ROUTER_FALLBACK_INTENT` 控制，默认 `rag` —— 贴近改造前行为，且检索为空时 prompt 已有「未检索到知识库内容」的谨慎分支。

### 5.3 检索（`rag/retrieval.py::retrieve`）

同步函数，调用方负责放进线程池。

```
embed_query(question)                         # bge 查询向量（带 query instruction）
   │
search(..., mode="auto")                      # 混合检索
   │  HYBRID：稠密 + BM25 → 服务端 RRFRanker
   │  失败自动降级为 DENSE 并记 warning
   ▼
outcome.nodes / outcome.similarities
   │
_unique_parent_ids → fetch_parents(一次 IN 查询)   # 父块正文从 SQLite 批量取
   │
按 parent_id 折叠去重 + 阈值过滤 + 字符预算裁剪
   ▼
sources: list[dict]
```

细节：

- **召回放大**：`child_top_k = min(rag_top_k × rag_child_top_k_multiplier, 50)`，先多召回子块，折叠后再截到 `rag_top_k`；
- **阈值语义分层**：hybrid 返回的是 RRF 融合分（量级约 0.0x），用 `RAG_HYBRID_MIN_SCORE`；dense 返回余弦相似度，用 `RAG_MIN_SCORE`。两者不可混用；
- **父块优先**：命中的子块 → 取其 `parent_id` → 用父块全文作为 prompt 上下文（`text`），子块原文截断 200 字作为前端展示用的 `matched_text`；
- **字符预算**：单条父块最多 `RAG_PARENT_MAX_CHARS`（1600），整轮上下文最多 `RAG_CONTEXT_MAX_CHARS`（8000），超出直接跳过；
- **旧数据兼容**：节点没有 `parent_id` 时按 `child_text[:200]` 去重、`RAG_CHUNK_MAX_CHARS` 截断；
- 父块读取失败**不阻断问答**，记 warning 后回退为子块文本。

`sources` 每项结构：

```json
{
  "index": 1, "node_id": "...", "parent_id": "1:13", "parent_index": 13,
  "file_id": 1, "file_name": "手册.pdf", "knowledge_base_id": 1,
  "page_label": "15", "chunk_index": 25,
  "score": 0.0295, "score_type": "rrf",
  "matched_text": "命中的子块原文…", "text": "父块全文"
}
```

### 5.4 Prompt 组装（`rag/prompt.py`）

两条分支共用同一套历史拼接逻辑 `_history_messages()`：

| 分支 | System Prompt | 内容 |
| --- | --- | --- |
| RAG | `SYSTEM_PROMPT` | 严格依据【知识库上下文】回答、句末标 `[1][2]`、上下文不足时明说 |
| 通用 | `GENERAL_SYSTEM_PROMPT` | 只用通用知识、禁止提及「知识库/检索/上下文」、禁止输出引用编号、**忽略历史里的引用格式** |

RAG 分支检索为空时，system 追加「本轮未检索到知识库内容，请基于常识谨慎回答，并说明未引用知识库」。

> 通用分支为什么不复用 RAG 的空来源分支：那段提示本身会引入「知识库」话术，与「常规问题」体验冲突。

### 5.5 SSE 协议

`text/event-stream; charset=utf-8`，每帧 `data: {json}\n\n`，结束帧 `data: [DONE]\n\n`。

| 事件 | 载荷 | 说明 |
| --- | --- | --- |
| `intent` | `{"intent": "rag" \| "general"}` | 本次路由结果（新增） |
| `sources` | `{"sources": [...]}` | general 时为 `[]`，事件始终发送 |
| `delta` | `{"content": "..."}` | 增量文本 |
| `done` | `{"conversation_id", "message_id"}` | 已落库 |
| `error` | `{"code", "message"}` | 生成期错误（检索期错误走 HTTP JSON） |

稳定性约束：

- 客户端断开检测 `await request.is_disconnected()` 在**每个 delta 前**检查，断开则不保存消息；
- `finally` 里只 `await stream.aclose()`，**不 yield**——客户端断开时 `GeneratorExit` 会因再次 yield 报错；
- 空回答发 `EMPTY_RESPONSE`，不落库。

### 5.6 消息与事务

`services/chat_service.py` 用**三个短事务**替代长连接：

| 阶段 | 函数 | 事务内容 |
| --- | --- | --- |
| A | `prepare_stream` | 校验会话/知识库、读历史、写 user 消息、首次提问时用前 30 字做标题 |
| — | （流式生成中） | 不持有数据库连接 |
| D | `save_assistant` | 写 assistant 消息、更新会话时间 |

历史读取按「从新到旧累加字符预算」截断到 `MEMORY_MAX_MESSAGES`（10 条）/ `MEMORY_MAX_CHARS`（4000），再反转为正序。

`sources` 落库语义：

| 情况 | 落库值 | 前端展示 |
| --- | --- | --- |
| 走 RAG 且命中 | 数组 | 引用来源面板 |
| 走 RAG 无命中 | `[]` | 无面板 |
| 常规问题 | `null` | 「未使用知识库」标签 |

> 意图没有单独落库：项目无迁移框架，加列不会自动生效。`sources === null` 已能可靠区分「常规回答」，旧数据的 `sources` 恒为数组，不会误标。

---

## 6. 前端架构

### 6.1 结构

```
main.ts → App.vue → router ( /chat, /knowledge )
                     ├─ ChatView      ── useChatStore
                     └─ KnowledgeView ── useKnowledgeStore

api/http.ts        fetch 封装、ApiError、统一错误体解析
api/chat.ts        REST + 手写 SSE 流解析
api/knowledge.ts   知识库/文件 CRUD 与上传
api/health.ts      健康检查
```

组件职责：

| 组件 | 职责 |
| --- | --- |
| `ConversationSidebar` | 会话列表、新建/重命名/删除 |
| `MessageBubble` | 单条消息、Markdown 渲染、来源面板、复制 |
| `StreamingBubble` | 流式中的气泡 + 状态文案（结合知识库 / 直接回答） |
| `SourcePanel` | 引用来源折叠面板（`rrf` 显示融合分、`dense` 显示相似度百分比） |
| `MarkdownContent` | `marked` + `dompurify` 安全渲染 |

### 6.2 SSE 消费

`api/chat.ts::streamChat` 用 `fetch` + `ReadableStream` 手写解析（不用 `EventSource`，因为需要 POST）：

- 按 `\n\n` 切帧，只处理 `data:` 行；
- 收到 `error` 帧转成 `ApiError` 抛出；
- `AbortController` 支持「停止生成」，中止后本轮不落库。

### 6.3 状态管理

Pinia 组合式 store（`stores/chat.ts`）持有：会话列表、消息列表、流式内容、流式来源、**流式意图**、错误。流结束后统一 `loadMessages` + `loadConversations`，以服务端数据为准。

---

## 7. 配置体系

`core/config.py::Settings`（pydantic-settings，`lru_cache` 单例，读 `.env`）：

| 分组 | 关键项 |
| --- | --- |
| 基础 | `APP_NAME` `DEBUG` `CORS_ORIGINS` |
| 存储 | `DATABASE_URL` `DATA_DIR` `UPLOAD_DIR` `MAX_UPLOAD_MB` |
| Milvus | `MILVUS_URI` `MILVUS_TOKEN` `MILVUS_COLLECTION` |
| Embedding | `EMBEDDING_PROVIDER` `EMBEDDING_MODEL_NAME` `EMBEDDING_DEVICE` `EMBEDDING_DIM` `EMBEDDING_QUERY_INSTRUCTION` |
| 对话模型 | `DEEPSEEK_API_KEY` `DEEPSEEK_MODEL_NAME` `DEEPSEEK_BASE_URL` `LLM_TEMPERATURE` |
| 意图识别 | `INTENT_ROUTING_ENABLED` `INTENT_ROUTER_TOOL_CHOICE` `INTENT_ROUTER_TIMEOUT_SECONDS` `INTENT_ROUTER_HISTORY_MESSAGES/CHARS` `INTENT_ROUTER_QUERY_MAX_CHARS` `INTENT_ROUTER_FALLBACK_INTENT` |
| 切分 | `CHUNK_SIZE` `CHUNK_OVERLAP` `PARENT_CHILD_ENABLED` `PARENT_CHUNK_SIZE` `PARENT_CHUNK_OVERLAP` `RAG_PARENT_MAX_CHARS` |
| 检索 | `HYBRID_RETRIEVAL_ENABLED` `RAG_RRF_K` `RAG_SPARSE_ANALYZER` `RAG_HYBRID_MIN_SCORE` `RAG_CHILD_TOP_K_MULTIPLIER` `RAG_TOP_K` `RAG_MIN_SCORE` `RAG_CHUNK_MAX_CHARS` `RAG_CONTEXT_MAX_CHARS` |
| 会话记忆 | `MEMORY_MAX_MESSAGES` `MEMORY_MAX_CHARS` |

> 注意：`RAG_*` 里的 `RAG_TOP_K` 等是**检索参数**，与意图路由的 `INTENT_ROUTER_*` 是两套独立配置。

---

## 8. 错误处理与可观测性

### 8.1 异常体系（`core/errors.py`）

`AppError` 基类携带 `status_code` / `code` / `message`，子类覆盖默认值：

```
AppError
├─ NotFoundError            404 NOT_FOUND
├─ InvalidArgumentError     400 INVALID_ARGUMENT
├─ InvalidFileTypeError     400 INVALID_FILE_TYPE
├─ InvalidDocumentError     400 INVALID_DOCUMENT
├─ FileTooLargeError        413 FILE_TOO_LARGE
├─ FileBusyError            409 FILE_BUSY
├─ MilvusUnavailableError   503 MILVUS_UNAVAILABLE
├─ EmbeddingUnavailableError 503 EMBEDDING_UNAVAILABLE
└─ LLMUnavailableError      503 LLM_UNAVAILABLE
```

统一响应体：

```json
{"error": {"code": "MILVUS_UNAVAILABLE", "message": "向量检索服务暂不可用"}}
```

注册三个处理器：`AppError`、`RequestValidationError`（转 400）、兜底 `Exception`（500 + `logger.exception`）。

**流式接口是例外**：进入生成器后无法再改 HTTP 状态码，因此 `chat_stream` 自己捕获 `AppError` / 未知异常，转成 SSE `error` 帧。

### 8.2 日志

- 全局 `logging.basicConfig(level=INFO)`，格式含时间、级别、模块名；
- 关键节点记 `logger.exception`：入库失败、流式失败；
- 意图路由记 `intent` / `has_query` / `decided_by`，兜底额外记原因——**观察 `decided_by="fallback"` 的比例即可判断模型是否支持函数调用**。

### 8.3 健康检查（`GET /api/health`）

| 检查项 | 方式 |
| --- | --- |
| 数据库 | `SELECT 1` |
| Milvus | 线程池 + 5s 超时 → `collection_capabilities()` 校验字段、维度、BM25、分词器 |
| Embedding | 单例是否已加载 |
| LLM | 是否配置 API Key |

Milvus 状态区分 `ok` / `degraded`（缺字段、维度不符、混合检索缺 BM25）/ `unavailable` / `timeout`。

---

## 9. 测试策略

`tests/` 共 48 个用例，全部不依赖真实 Milvus / DeepSeek：

| 文件 | 覆盖内容 |
| --- | --- |
| `test_retrieval.py` | 父子折叠、阈值、top_k、字符预算、旧数据兼容 |
| `test_vector_store_hybrid.py` | BM25/RRF 配置、hybrid 降级、schema 校验 |
| `test_parent_child_split.py` | 父子切分与 metadata |
| `test_parent_store.py` | 父块读写与幂等替换 |
| `test_ingestion_metadata.py` | 入库元数据 |
| `test_ingest_service.py` | 后台任务成功/失败/状态恢复 |
| `test_prompt.py` | 两种 Prompt 组装 |
| `test_router.py` | 意图路由 14 例：正常/未知工具/异常/超时/禁用/截断/raw tool_calls |
| `test_chat_stream.py` | 流式编排 6 例：分流、跳过检索、错误帧、断开、空回答 |

测试手法：

- 统一 `monkeypatch` 替换模块级符号（`embed_query`、`search`、`get_chat_model` 等）；
- 配置用 `monkeypatch.setattr(get_settings(), "字段", 值)`；
- 异步逻辑用 `asyncio.run(...)` 包装（未引入 pytest-asyncio）；
- 路由层测试直接 `await chat_stream(payload, FakeRequest())` 并遍历 `response.body_iterator`，不启动 FastAPI。

---

## 10. 运维脚本（`tools/`）

| 脚本 | 用途 |
| --- | --- |
| `smoke_search.py` | 分别跑 dense / sparse / hybrid 对比召回；确认 hybrid 分数是 0.0x 量级（RRF）而非余弦值 |
| `reingest_parent_child.py` | 把已有文件重新导入父子切片 + 混合检索链路（支持 `--dry-run`、`--include-failed`） |

用法示例：

```bash
python -m know_answer.tools.smoke_search --query "某个精确关键词" --top-k 5
python -m know_answer.tools.reingest_parent_child --dry-run
```

---

## 11. 关键设计决策汇总

| 决策 | 原因 |
| --- | --- |
| 父子两级切分 | 子块保证检索精度，父块保证上下文完整，避免「切得太碎答不全」 |
| 父块只存 SQLite | 省向量库存储与检索开销；回填只需一次 `IN` 查询 |
| 服务端 RRF 融合 | Milvus 原生 `RRFRanker` 一次请求完成稠密 + 稀疏融合，少一次网络往返 |
| hybrid 失败降级 dense | 老 collection / 服务端不支持时功能不中断 |
| 入库单线程串行 | CPU 上的本地 embedding 并发只会互相抢占 |
| 模块级单例 + 双检锁 | 模型加载昂贵；FastAPI 依赖注入体系不适合管理重对象生命周期 |
| 路由在流式之前完成 | 检索失败仍可返回标准 JSON 503；进入 SSE 后错误处理路径会分叉 |
| 意图用 Tool Calling 而非关键词规则 | 语义泛化好，能处理同义表达与多轮指代；同时天然支持 query 改写 |
| 意图不落库 | 无迁移框架；`sources === null` 已足够表达「未使用知识库」 |
| 通用分支独立 system prompt | 避免 RAG 话术（引用编号、「知识库」措辞）泄漏到闲聊场景 |
| 无登录鉴权 | 定位为本地/内网工具，README 明确要求只监听 `127.0.0.1` |

---

## 12. 已知限制与扩展点

### 12.1 限制

1. **无数据库迁移**：新增表字段需要手工 `ALTER TABLE` 或删库重建。
2. **无鉴权/多租户**：任何能访问端口的人都能读写全部知识库。
3. **入库任务不持久化**：进程重启后 `pending/processing` 直接标记失败，需重新上传。
4. **单进程假设**：所有组件是进程内单例，多 worker 部署会各自加载一份模型；单线程入库队列也不跨进程。
5. **`models/` 目录是历史遗留**：`chat_models.py` / `embed_models.py` 只是早期 LangChain 封装示例，主链路未引用。
6. **意图路由多一次 LLM 往返**：首字延迟约增加 0.5–2s；`INTENT_ROUTING_ENABLED=false` 可一键关闭。
7. **`deepseek-flash` 的函数调用能力需实测**：若端点不支持 `bind_tools`，会静默兜底为「全部走 RAG」，通过日志 `decided_by="fallback"` 可发现。

### 12.2 可扩展方向

| 方向 | 接入点 |
| --- | --- |
| 换向量库 | `rag/vector_store.py`（实现 LlamaIndex `VectorStore` 接口） |
| 换对话模型 | `rag/llm.py::get_chat_model`（`init_chat_model` 支持多 provider） |
| 增加意图类别 | `rag/router.py` 增加 `@tool` 与 `IntentDecision.intent` 分支 |
| 查询改写/多轮检索 | `rag/router.py` 已返回改写 query；可扩展为 Agent 循环 |
| 重排序（rerank） | `rag/retrieval.py` 在折叠父块后插入 rerank |
| 流式持久化意图 | 给 `messages` 表加列并补轻量迁移 |

---

## 附录 A：API 一览

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/knowledge-bases` | 创建知识库 |
| GET | `/api/knowledge-bases` | 知识库列表 |
| GET/PATCH/DELETE | `/api/knowledge-bases/{id}` | 详情 / 更新 / 删除 |
| POST | `/api/knowledge-bases/{id}/files` | 上传文件（202，后台入库） |
| GET | `/api/knowledge-bases/{id}/files` | 文件列表 |
| GET/DELETE | `/api/files/{id}` | 文件状态 / 删除 |
| POST | `/api/conversations` | 创建会话 |
| GET | `/api/conversations` | 会话列表 |
| PATCH/DELETE | `/api/conversations/{id}` | 重命名 / 删除 |
| GET | `/api/conversations/{id}/messages` | 历史消息 |
| POST | `/api/chat/stream` | 流式问答（SSE） |

## 附录 B：关键文件速查

| 关注点 | 文件 |
| --- | --- |
| 应用启动 / 预热 / SPA 托管 | `src/know_answer/app/_app.py` |
| 流式问答编排 | `src/know_answer/app/routes/chat.py::chat_stream` |
| 意图识别 | `src/know_answer/rag/router.py::classify_intent` |
| 检索与父块回填 | `src/know_answer/rag/retrieval.py::retrieve` |
| Milvus 适配与降级 | `src/know_answer/rag/vector_store.py::search` |
| Prompt | `src/know_answer/rag/prompt.py` |
| 入库解析切分 | `src/know_answer/data_index/doc_loading.py::DocumentIngestor` |
| 后台入库任务 | `src/know_answer/services/ingest_service.py::process_file` |
| 数据模型 | `src/know_answer/db/models.py` |
| 配置 | `src/know_answer/core/config.py` |
| 前端 SSE 解析 | `frontend/src/api/chat.ts::streamChat` |
| 前端状态 | `frontend/src/stores/chat.ts` |

## 附录 C：本地启动

```bash
# 后端
uv sync
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
uv run uvicorn know_answer.app._app:app --reload --port 8000

# 前端（开发模式，/api 由 Vite 代理）
cd frontend && npm install && npm run dev

# 生产：构建后由后端直接托管
cd frontend && npm run build
# 然后访问 http://127.0.0.1:8000/chat

# 测试
uv run pytest -q
```
