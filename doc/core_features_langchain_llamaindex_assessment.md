# 核心功能清单与 LangChain + LlamaIndex 重构适配性评估

> 分析对象：`advanced-rag-qa` 当前工作区（分支 `develop-1.0`）。
> 分析日期：2026-11-04。
> 分析范围：`backend/` 后端主链路、`src/` 前端交互与 SSE 契约、`backend/config.py`、`backend/model/models.py`。
> 本文只做功能提炼与重构适配性判断，**不包含任何实际重构动作、代码修改或依赖升级**。

---

## 0. 结论先行

1. **不建议用 LangChain + LlamaIndex 整体重写本项目。** 本项目的核心价值不在「调用模型」，而在确定性的检索编排、双重数据隔离、逐路失败降级、SSE/Trace 协议和附件生命周期管理，这些是通用框架不直接提供、也不应被框架默认行为覆盖的部分。
2. **适合替换的是「能力适配层」与「索引/检索组合层」**：文档解析、Node 切分、Embedding、向量存储、混合召回、RRF、reranker、Memory 组装、RAGAS 评估接入。
3. **必须保留自研的是「业务编排层」与「契约层」**：鉴权与多租户、MySQL 事务与回滚、`knowledge_base_id` 双隔离、BM25 的 SQL 预筛与内存上界、SSE 帧与 Trace 事件 schema、附件登记/消费/墓碑/清扫、键集分页、限流与启动守卫。

当前依赖现状：`langchain-openai==1.6.7` 只被当作 OpenAI-compatible `ChatOpenAI` 客户端使用；**LlamaIndex 尚未引入**；Embedding、rerank、BM25、RRF、父子切片、记忆、RAGAS 编排均为自研代码。

### 适配性图例

| 标记 | 含义 |
| --- | --- |
| ✅ 适合 | 框架有成熟抽象，替换后能减少维护成本，现有契约可平移；仍需保留少量适配代码。 |
| 🟡 部分适合 | 框架可承担其中一部分；关键行为（降级、隔离、确定性兜底、协议、事务）必须自研或保留。 |
| ❌ 不适合 | 属于业务、安全、协议或工程保障层，框架没有对应抽象；强行迁移只会增加耦合。 |

---

## 1. 核心功能总览

| 编号 | 功能 | 优先级 | 适配性 | 一句话判断 |
| --- | --- | --- | --- | --- |
| F01 | 用户认证、令牌吊销与访问控制 | P0 | ❌ | 安全边界，框架不负责。 |
| F02 | 知识库多租户隔离与生命周期 | P0 | ❌ | 业务与权限模型，必须保留。 |
| F03 | 知识文件上传、解析与入库事务 | P0 | 🟡 | 解析可用框架，事务编排保留。 |
| F04 | 语义分块、结构元数据与父子切片 | P0 | 🟡 | NodeParser 可承接，中文结构识别需自研。 |
| F05 | Embedding 与 Milvus 向量索引 | P0 | ✅ | LangChain Embeddings + LlamaIndex VectorStore 可直接承接。 |
| F06 | RAG 路由与查询规划 | P0 | ✅ | structured output + HyDE/改写可承接，保留兜底。 |
| F07 | 多路向量召回与关键词/BM25 召回 | P0 | 🟡 | 混合检索可承接，SQL 预筛与内存上界需保留。 |
| F08 | RRF 融合、rerank 与最终上下文选择 | P0 | 🟡 | 融合/重排可承接，关键词补位逻辑需保留。 |
| F09 | 父子切片聚合与上下文预算 | P0 | ✅ | AutoMerging/ParentDocumentRetriever 可承接。 |
| F10 | SSE 流式回答生成与后备模型接管 | P0 | 🟡 | 模型流可承接，SSE 帧与 reset 语义需保留。 |
| F11 | 会话记忆（短期窗口 + 长期摘要） | P1 | 🟡 | Memory 抽象可承接，持久化与触发时机保留。 |
| F12 | 图片附件问答与视觉理解 | P1 | 🟡 | 多模态消息可承接，OSS 生命周期需保留。 |
| F13 | 聊天附件生命周期管理 | P1 | ❌ | 状态机 + 清扫任务，框架无对应抽象。 |
| F14 | 会话/消息管理、导出与反馈 | P1 | ❌ | 业务 CRUD 与删除语义，必须保留。 |
| F15 | 检索 Trace 与可观测性 | P1 | 🟡 | Callback/Instrumentation 可采集，事件 schema 需保留。 |
| F16 | RAGAS 在线评估 | P1 | ✅ | RAGAS 本身可与 LangChain/LlamaIndex 集成。 |
| F17 | 回答 groundedness 轻量校验 | P1 | ❌ | 确定性词法校验，框架无等价物。 |
| F18 | 键集分页与列表查询 | P2 | ❌ | 数据库访问策略，与 RAG 框架无关。 |
| F19 | 限流、并发闸门与启动守卫 | P2 | ❌ | 进程级工程保障，必须保留。 |
| F20 | 前端交互与 SSE 协议 | P2 | ❌ | 不在 LangChain/LlamaIndex 重构范围；协议必须保持兼容。 |

统计：**20 项核心功能**；其中 ✅ 适合 4 项、🟡 部分适合 8 项、❌ 不适合 8 项。

---

## 2. 详细功能清单

## P0：核心问答闭环

### F01 用户认证、令牌吊销与访问控制

- **功能名称**：JWT 登录、登出、改密、令牌世代与单 token 吊销、登录失败节流。
- **用途/目标**：建立请求身份，保证后续所有知识库、会话、消息、附件操作都绑定到正确的 `user_id`；改密后旧 token 立即失效，登出只影响当前会话。
- **关键输入与输出**：
  - 输入：`username` / `password`；后续请求头 `Authorization: Bearer <token>`。
  - 输出：`LoginResponse(token, username)`；鉴权成功得到 `User`；失败返回 401/429。
- **主要流程 / 依赖模块**：
  - `backend/service/auth_service.py::login` → `bcrypt` 校验 → `create_token(sub, ver, jti, exp)`。
  - `authenticate` 统一链路：验签 → 回查用户 → 校验 `token_version` → 查 `revoked_tokens.jti`。
  - `service/rate_limit.py::FailureWindow` 做「账号 + 来源地址」滑动窗口登录节流。
  - 所有 FastAPI 依赖入口走 `get_current_user`；SSE 入口 `stream_chat` 也复用同一条 `authenticate`。
- **现状边界与待确认点**：
  - 使用 HS256 共享密钥，`SECRET_KEY` 为占位值时拒绝启动；无 refresh token、无 OIDC/SSO。
  - 登录失败计数在**进程内存**，多 worker/多实例时真实上限 = 配置值 × 进程数。
  - 未读取 `X-Forwarded-For`，反向代理后按代理地址限流，只会更严。
  - 待确认：是否需要接入企业 SSO / LDAP / OAuth2；是否需要跨进程共享限流状态。
- **重构适配性**：❌ 不适合。
- **理由**：LangChain 与 LlamaIndex 都不提供认证、鉴权、令牌吊销和租户边界；把这块交给框架只会把安全边界埋进第三方抽象。建议原样保留，最多把「当前用户」作为参数显式传入新的检索/生成适配层。

---

### F02 知识库多租户隔离与生命周期

- **功能名称**：知识库创建、重命名、删除、默认库兜底、会话绑定，以及 MySQL 元数据 + Milvus 向量的双重隔离。
- **用途/目标**：让每个用户只能看到并使用自己的知识库；同一知识库 ID 同时约束关系库记录与向量记录，避免跨库检索和跨用户泄露。
- **关键输入与输出**：
  - 输入：`user_id`、`knowledge_base_id`（可空）、知识库名称。
  - 输出：知识库列表/详情；删除时返回 `fallback_knowledge_base_id`；越权或不存在统一 404。
- **主要流程 / 依赖模块**：
  - `backend/service/knowledge_service.py`：`list_knowledge_bases`、`create_knowledge_base`、`rename_knowledge_base`、`delete_knowledge_base`。
  - `backend/crud/knowledge_base.py`：按 `user_id` 过滤；删除时先把该库文件逐个删向量，再删文件与知识库行，最后把本人会话改绑到兜底库、他人悬挂引用置空。
  - 检索侧 `query_vectors(..., knowledge_base_id=...)` 生成 `knowledge_base_id == N` 过滤表达式；关键词召回同样按库过滤。
  - 会话创建时绑定知识库 `Conversation.knowledge_base_id`，后续对话复用绑定，避免中途串库。
- **现状边界与待确认点**：
  - 历史数据的 `user_id` 为 NULL 时对所有人不可见；需要回填脚本处理。
  - 至少保留一个知识库；删除知识库按文件逐个清理向量，没有批量删除接口。
  - 会话行只保存一个知识库绑定，不支持单会话跨库检索。
  - 待确认：是否需要组织/部门级共享知识库；是否需要「一个会话挂多个知识库」；是否需要软删除与恢复。
- **重构适配性**：❌ 不适合。
- **理由**：这是业务权限模型，不是检索框架能力。LlamaIndex 的 `VectorStoreIndex` 只能接受一个过滤条件，无法替代「归属校验 + 会话绑定 + 兜底库 + 删除改绑」这套业务规则。建议保留为外层服务，把 `knowledge_base_id` 作为必传的 metadata filter 传给新检索层。

---

### F03 知识文件上传、解析与入库事务

- **功能名称**：TXT / Markdown / DOCX / PDF 上传、文本抽取、元数据落库、分块与向量写入的失败回滚。
- **用途/目标**：把企业文档变成可检索的知识资产；任一步失败都不能留下「列表里有、检索不到」或「向量残留、文件已删」的半成品。
- **关键输入与输出**：
  - 输入：`multipart/form-data` 文件、可选 `knowledge_base_id`、`Content-Length`。
  - 输出：`KnowledgeFile` 记录（id、名称、大小、正文）；HTTP 400/500；成功后触发该知识库 BM25 预热。
- **主要流程 / 依赖模块**：
  - `backend/service/knowledge_service.py::upload_knowledge`。
  - 类型白名单 `resolve_knowledge_upload_type`（`.txt/.md/.docx/.pdf`，MIME 与扩展名双校验）。
  - 大小限制：`KNOWLEDGE_UPLOAD_MAX_MB`（默认 10MB），预检 + 分块读取二次核验。
  - `crud/knowledge_file.py::extract_file_text`：DOCX 用 `python-docx` 按块抽取，PDF 用 `pypdf`。
  - 空文本直接 400（扫描件/图片型 PDF 没有 OCR，不会被静默入库）。
  - 入库顺序：写 `KnowledgeFile` → 分块 → 写 `knowledge_chunk_metadata` → `add_chunks` 写向量；失败按「向量是否确认清理成功」决定删行还是保留行等待重建。
  - 所有解析/分块/向量化同步工作通过 `run_ingest_step` 进入专用线程池，共享总超时（默认 300s）。
- **现状边界与待确认点**：
  - 文件正文存 MySQL `LONGTEXT`，不落对象存储；大文件会放大数据库体积。
  - 无 OCR、无表格结构化解析、无图片内文字提取；扫描 PDF 会被拒绝。
  - 单请求单文件，无去重、无版本、无增量更新；重传即新文件。
  - 入库超时后清理挂在线程池完成回调，属于「尽力而为」。
  - 待确认：是否要支持 DOC/PPT/XLSX/HTML；是否需要 OCR；是否要把正文迁到对象存储；是否需要内容哈希去重。
- **重构适配性**：🟡 部分适合。
- **理由**：LlamaIndex 的 `SimpleDirectoryReader` / `DocxReader` / `PDFReader`、`IngestionPipeline` 可以替换解析与「解析→切分→Embedding→写入」的流水线，减少样板代码。但本项目的事务回滚、失败清理、空文本拒绝、入库超时、BM25 预热是业务语义，必须保留在外层编排中。建议只替换「抽取器」和「流水线节点」，不替换上传服务本身。

---

### F04 语义分块、结构元数据与父子切片

- **功能名称**：按标题层级 / 段落 / 句子 / 表格切分，token 预算控制，章节与条款元数据，父子切片构建。
- **用途/目标**：让切片既适合向量召回（child 粒度小），又保留回答所需的完整上下文（parent 粒度大）；同时把章节路径、条款编号、表格行列等结构信息写入 MySQL，供检索后聚合与前端展示。
- **关键输入与输出**：
  - 输入：抽取后的全文、`file_id`、`file_name`、`CHUNK_TOKEN_*` 配置。
  - 输出：`(children, metadata_chunks)`；每个 child 含 `id/text/metadata`，metadata 含 `section_path`、`article_no`、`token_count`、`table_id/row_no`、`parent_id`、`context_prefix` 等；parent 行含 `parent_text` 与 `child_chunk_ids`。
- **主要流程 / 依赖模块**：
  - `backend/crud/knowledge_file.py::chunk_text`：块切分 → `_semantic_units` → 标题识别、表格行识别、长段回退 → token 硬切（切点必须落在 UTF-8 字符边界）。
  - `build_parent_child_chunks`：按章节/长度分组 → 生成确定性上下文前缀 → child 超硬上限时二次拆分 → parent 只落 MySQL，不进 Milvus。
  - `resolve_chunk_token_budget`：默认目标 400 token、硬上限 `500 × 0.85 = 425`、重叠 50。
  - `KnowledgeChunkMetadata` 表持久化结构信息；`replace_chunk_metadata` 整体替换。
- **现状边界与待确认点**：
  - 父子切片默认关闭（`PARENT_CHILD_ENABLED=false`），开启后 parent 只存在 MySQL，Milvus schema 不变。
  - 标题/条款/表格识别是中文规则启发式，不是通用文档结构解析；表格行超预算时保持原子性。
  - token 计数依赖 `tiktoken cl100k_base`，不可用时退化为 UTF-8 字节上界。
  - 分块覆盖率低于 50% 只告警，不阻断。
  - 待确认：是否需要按业务文档类型定制解析器；是否需要语义切分（embedding-based）；是否需要保留表格结构为 Markdown/HTML；是否需要多语言分词。
- **重构适配性**：🟡 部分适合。
- **理由**：LlamaIndex 的 `NodeParser` / `HierarchicalNodeParser` / `SentenceSplitter` 提供了成熟的节点抽象、元数据挂载和父子关系，可作为主要承载。但本项目的标题层级、条款编号、表格行列、UTF-8 字符边界硬切是中文企业文档特有的规则，需要自定义 NodeParser 或保留现有切分器并包装成 LlamaIndex `Node`。建议「用 LlamaIndex 的 Node/关系模型，保留现有切分算法」。

---

### F05 Embedding 与 Milvus 向量索引

- **功能名称**：文本向量化、批量与重试、Milvus collection 管理、按知识库过滤的向量写入与检索。
- **用途/目标**：建立和维护语义索引；在知识库隔离前提下召回语义相关切片。
- **关键输入与输出**：
  - 写入输入：child chunks、`file_id`、`file_name`、`knowledge_base_id`。
  - 查询输入：query 文本、`top_k`、`knowledge_base_id`。
  - 输出：命中 chunk（`id/chunk_id/content/file_name/file_id/route/distance`）。
- **主要流程 / 依赖模块**：
  - `backend/rag/milvus_client.py`：`_OpenAICompatibleEmbeddingFunction` 调用 OpenAI-compatible `/embeddings`（默认 DashScope `text-embedding-v4`，1024 维）。
  - `_embed_documents` 按条数 / token 预算分批，429/5xx 按 `Retry-After` 与上限退避；入库有总 deadline。
  - `add_chunks` 采用「先取全部向量，再删旧 file_id 向量，最后插入」的替换语义，向量化失败时旧索引保持原样。
  - `query_vectors` 使用 `AUTOINDEX` + `COSINE`，`limit = min(max(top_k,1),20)`，filter 为 `knowledge_base_id == N`。
  - Milvus Lite 为进程内单例并加初始化锁，避免同一数据目录多实例锁冲突。
- **现状边界与待确认点**：
  - collection schema 固定：`id/embedding/content/file_id/file_name/chunk_id/knowledge_base_id`；没有章节、条款、parent 等字段，结构信息只在 MySQL。
  - 未配置 Embedding 时会使用**本地哈希向量**（仅开发用途）；配置后调用失败则抛 `EmbeddingBackendError`，不会退化为哈希向量。
  - 没有多租户物理分区，隔离依赖 filter；没有向量删除审计、没有索引重建任务队列。
  - 待确认：是否继续使用 Milvus；是否需要换成其他向量库；是否允许修改 collection schema；数据规模与并发量。
- **重构适配性**：✅ 适合。
- **理由**：LangChain 的 `Embeddings` 接口和 LlamaIndex 的 `MilvusVectorStore` 能直接承接 Embedding 调用与向量读写，还能复用其批处理、异步与元数据过滤能力。需要保留的是：按知识库过滤、替换语义、429/5xx 重试与 deadline、Lite 单例锁、失败不降级为哈希向量。建议把 `milvus_client.py` 收缩为「LangChain Embeddings + LlamaIndex VectorStore 的适配器」，不要直接把业务代码散落到框架 API。

---

### F06 RAG 路由与查询规划

- **功能名称**：判断本轮是否需要检索；生成 HyDE 文档、问题改写、子问题、关键词与必需证据。
- **用途/目标**：避免所有问题都强制检索（数学、闲聊、翻译走直答）；把口语化问题扩展成更适合召回的多路查询。
- **关键输入与输出**：
  - 输入：用户问题、会话记忆（裁剪到 1800 字符）、知识库名称、附件数量。
  - 输出：`{need_rag, route, confidence, reason, source}`；`{hyde_document, rewrites, sub_questions, keywords, required_evidence, original_question, simplified_question}`。
- **主要流程 / 依赖模块**：
  - `backend/rag/retrieval.py::decide_need_rag` → `rag.llm.call_router_json`（router 阶段模型）→ `_normalize_decision`；置信度低于 0.55 或 JSON 无效时保守进入 RAG。
  - `build_query_plan` → `call_chat_json(stage="planner")` → 失败时用 `_fallback_keywords` 做确定性兜底。
  - `_build_route_specs` 把 `planned/simplified/sub_question_*/hyde/rewrite_*` 转成多路向量查询，并去重。
- **现状边界与待确认点**：
  - 路由和规划是两次独立 LLM 调用，串行；没有并行化，也没有缓存。
  - 规划字段是固定 JSON schema，模型返回异常时走兜底而不是重试。
  - 路由只决定「是否检索」，不决定「检索哪个知识库」；知识库由请求参数/会话绑定决定。
  - 待确认：是否需要多知识库路由；是否需要把路由与规划合并为一次调用；是否需要可配置的直答白名单。
- **重构适配性**：✅ 适合。
- **理由**：LangChain 的 structured output / `with_structured_output`、LlamaIndex 的 `HyDEQueryTransform`、`MultiStepQueryEngine`、`SubQuestionQueryEngine` 都能承接这些模式。建议替换「Prompt + JSON 解析 + 模型调用」这一层，但保留：低置信度保守进入 RAG、解析失败确定性兜底、固定字段契约、Trace 事件。

---

### F07 多路向量召回与关键词/BM25 召回

- **功能名称**：多查询向量召回、关键词召回、BM25 排序、单路失败降级、语料缓存与预热。
- **用途/目标**：同时覆盖语义相似与精确词面匹配（条款号、错误码、金额、专有名词）；某一路失败时只降级该路，不拖垮整轮问答。
- **关键输入与输出**：
  - 输入：检索问题、知识库 ID、`RETRIEVAL_ROUTE_TOP_K`（默认 8）、查询规划。
  - 输出：每路 `(route, chunks)` 列表；关键词候选带 `keyword_score`、`bm25_score`、`bm25_rank`。
- **主要流程 / 依赖模块**：
  - `retrieve_knowledge` 用 `asyncio.gather(..., return_exceptions=True)` 并发执行多个 `query_vectors` 和 `keyword_recall`。
  - 向量路遇到 `EmbeddingBackendError` 时跳过该路并记录 `embedding_error`；其他异常照常上抛。
  - 关键词路 `keyword_recall`：SQL `LIKE` 预筛 → 按主键游标分批（`LIMIT`）取正文 → 整篇归一化 → 窗口区间统计 → BM25 排序。
  - `rag/bm25.py` 纯 Python BM25（CJK unigram + bigram，ASCII 词与 `-_.` 组合词），语料按知识库缓存（LRU 8），支持启动/上传后后台预热。
- **现状边界与待确认点**：
  - BM25 语料上限 20,000 窗口 / 200,000 词项；超限降级为旧版词频打分，并在 Trace 标记。
  - 关键词召回有 SQL 预筛，成本与命中内容量相关而非知识库总量；含 `İ`/`Σ` 的文件回退逐窗口扫描。
  - 没有外部搜索、没有图检索、没有按 metadata 的结构化过滤（除知识库 ID）。
  - 多路数量由规划结果决定，未做动态预算控制。
  - 待确认：数据规模、中文分词是否要引入词典、是否需要保留「与旧实现逐条等价」的排序结果。
- **重构适配性**：🟡 部分适合。
- **理由**：LlamaIndex 的 `QueryFusionRetriever`、`VectorIndexRetriever`、`BM25Retriever` 和 LangChain 的 `EnsembleRetriever` 可以覆盖「多路召回 + 混合检索」的主干。但本项目关键词路的 SQL `LIKE` 预筛、分批游标、内存上界、BM25 语料缓存/预热、Unicode 归一化等价性、逐路降级都是为生产规模写的，框架默认实现不具备。建议：向量多路交给 LlamaIndex，关键词路保留现有实现并包装成自定义 Retriever。

---

### F08 RRF 融合、rerank 与最终上下文选择

- **功能名称**：RRF 融合、DashScope `qwen3-rerank` 重排、LLM 重排兜底、关键词最佳候选补位、去重与 top-N 截断。
- **用途/目标**：把多路召回结果合成一个稳定排序；用交叉编码器/重排模型提升相关性；确保「查询关键词强命中」的候选不会被重排模型漏掉。
- **关键输入与输出**：
  - 输入：`route_results`、原始问题、`RETRIEVAL_RERANK_TOP_N`（默认 5）。
  - 输出：最终 chunk 列表（带 `rrf_score`、`rerank_score`、`rerank_reason`、`routes`）与 rerank Trace。
- **主要流程 / 依赖模块**：
  - `rrf_fuse(route_results, k=60)`：按 `chunk_key` 聚合，跨分块方案的同文候选合并，累加 `1/(k+rank)`。
  - `rerank_chunks`：取 RRF 前 12 条 → `qwen3-rerank`（文档截断 4000 字符）→ 失败且开关允许时走 LLM JSON 重排 → 都失败则返回空列表并由调用方回退到 `fused`。
  - `select_final_chunks`：先去重再截断到 top-N；若关键词最佳候选 `keyword_score >= 10` 且命中查询关键词、且未被选中，则插到最前。
- **现状边界与待确认点**：
  - RRF 是无权重融合，各路等权；没有按路由类型加权。
  - 重排主服务是 DashScope `qwen3-rerank`，LLM 兜底输出 JSON 评分；两者失败语义不同。
  - 关键词候选可以绕过 rerank 直接进入最终上下文，这是刻意保留的「精确词面」保险。
  - 待确认：是否允许调整 RRF 权重 / top-N / 候选截断；是否要引入其他 reranker；是否需要把补位规则配置化。
- **重构适配性**：🟡 部分适合。
- **理由**：LlamaIndex 的 `QueryFusionRetriever` 已内置 RRF，`SentenceTransformerRerank` / `CohereRerank` 等 reranker 模块可替换主重排；LangChain 的 `ContextualCompressionRetriever` 也能组合。但 `select_final_chunks` 的关键词补位、去重键（分块方案命名空间）、失败回退到 `fused` 是业务规则，框架不会自动做。建议融合与重排用框架，最终选择保留自研。

---

### F09 父子切片聚合与上下文预算

- **功能名称**：child 命中后回查 parent、同一 parent 合并为一个上下文槽位、按 parent 边界裁剪 token 预算。
- **用途/目标**：召回时用细粒度 child 保证精度，生成时用完整 parent 保留但书、例外和上下文；避免同一 parent 的多个 child 重复占用上下文配额。
- **关键输入与输出**：
  - 输入：最终选择的 child chunks、MySQL `knowledge_chunk_metadata`。
  - 输出：展开后的 chunks（`content` 替换为 `parent_text`，附带 `parent_id`、`parent_section_path`、`parent_article_no`）、统计信息。
- **主要流程 / 依赖模块**：
  - `backend/rag/retrieval.py::expand_parent_chunks` → `_expand_parent_chunks_sync`。
  - 按 `(file_id, chunk_id)` 查 child 元数据 → 按 `parent_id` 查 parent 行 → 同一 parent 只保留一次 → 用 `parent_text` 替换 child 内容。
  - 关键词窗口来自原文偏移，不属于入库 child，按内容降级、不强行映射。
  - `_apply_parent_context_budget` 超 `PARENT_CONTEXT_MAX_TOKENS`（默认 6000）时从尾部整块丢弃，至少保留第一个槽位。
- **现状边界与待确认点**：
  - 默认关闭（`PARENT_CHILD_ENABLED=false`）；关闭时原样返回 child。
  - 旧文件没有元数据行时降级为 child；parent 行缺失/为空同样降级。
  - 只按 parent 边界裁剪，不在 parent 内部截断；单个 parent 超限时保留并标记 truncated。
  - 待确认：是否默认开启；parent 分组策略（按章节还是按 token）是否满足业务；是否需要把 parent 也写入向量库。
- **重构适配性**：✅ 适合。
- **理由**：LlamaIndex 的 `AutoMergingRetriever` / `ParentDocumentRetriever` / `HierarchicalNodeParser` 正是为这个模式设计的，能承接「child 检索、parent 返回」的主干。需要保留的是：MySQL 元数据表作为 parent 事实来源、关键词候选不映射、token 预算裁剪与 Trace 统计。建议用框架的节点关系与自动合并能力，替换手写回查与合并。

---

### F10 SSE 流式回答生成与后备模型接管

- **功能名称**：SSE 流式输出、回答 Prompt 组装、内容过滤识别、主模型失败后切换文本后备模型、`reset` 事件、成功才落库。
- **用途/目标**：让用户尽快看到增量回答；上游模型故障时仍能给出完整答案；失败内容不写成正式 assistant 消息。
- **关键输入与输出**：
  - 输入：`effective_question`、知识库上下文、会话记忆、`use_rag`。
  - 输出：SSE 文本 chunk、`sources`、`conversation`、`image_analysis`、`trace`、`reset`、`error`、`[DONE]`；成功时写 assistant 消息。
- **主要流程 / 依赖模块**：
  - `backend/rag/chains.py::stream_rag_answer` → `rag/llm.py::stream_answer_events`。
  - `build_answer_messages` 按 RAG / 直答两种模式组装 system + user prompt，明确「知识库上下文 > 当前问题 > 会话记忆」。
  - `get_stage_model("answer")` 取 DeepSeek 系列模型，`ChatOpenAI.astream` 流式输出。
  - 内容过滤：读 `finish_reason` 显式转 error，不切换后备模型。
  - 传输/配置失败：若 `TEXT_FALLBACK_ENABLED` 且已配置 key，先发 `reset` 作废已输出内容，再用后备模型从头生成。
  - `chat_service.stream_chat` 负责 SSE 帧编排、assistant 落库、RAGAS 调度、记忆摘要调度、Trace 收尾、并发槽归还。
- **现状边界与待确认点**：
  - 两套模型来源：DeepSeek 主模型 + DashScope/OpenAI-compatible 文本后备。
  - 没有 Agent / Tool 调用；生成阶段只做「Prompt → 模型 → 流」。
  - 失败不保存 assistant 消息；空正文按失败处理；客户端断开时靠 `finally` 归还并发槽并写 Trace 终态。
  - 待确认：是否需要保留 `reset` 协议；是否允许更换主/备模型；是否需要 token 级 usage 统计。
- **重构适配性**：🟡 部分适合。
- **理由**：LangChain 的 `astream_events` / `ChatOpenAI` 和 LlamaIndex 的 `astream_chat` 能承接模型流、重试与回调，但本项目 SSE 帧是前端和测试共同依赖的协议，`reset` 语义、内容过滤判定、失败不落库、并发槽生命周期都必须保留。建议只替换「模型调用与流式适配」，SSE 编排留在 `chat_service`。

---

## P1：业务支撑能力

### F11 会话记忆（短期窗口 + 长期摘要）

- **功能名称**：短期滑窗、长期摘要、近期上下文压缩、长期摘要二次压缩、记忆在检索/路由/生成三处的差异化使用。
- **用途/目标**：支持多轮追问中的指代消解、任务延续和用户偏好；同时控制 prompt 长度，避免历史无限增长。
- **关键输入与输出**：
  - 输入：`conversation_id`、当前用户消息 id、`memory_summary_upto_message_id`。
  - 输出：`memory_context`（长期摘要 + 最近窗口）、`retrieval_question`（带记忆的检索问题）。
- **主要流程 / 依赖模块**：
  - `backend/rag/memory_service.py`。
  - 短期：读取摘要指针之后、当前消息之前的最近 `MEMORY_WINDOW_TURNS`（默认 4）轮；超过 `MEMORY_RECENT_MAX_CHARS`（8000）时用 memory 阶段模型压缩，失败用确定性裁剪兜底。
  - 长期：assistant 消息保存成功后异步把滑出窗口的轮次合并进 `conversations.memory_summary`，推进 `memory_summary_upto_message_id`；超过 `MEMORY_SUMMARY_MAX_CHARS`（15000）做二次摘要。
  - 使用：检索问题最多 1500 字符（只用于指代消解）、路由最多 1800 字符、最终回答使用完整 `memory_context`。
  - 消息删除若覆盖摘要指针，直接作废摘要并重置指针。
- **现状边界与待确认点**：
  - 异步旁路、尽力而为；失败只记日志与 Trace，不影响主回答。
  - 没有向量记忆、没有跨会话用户画像、没有外部记忆服务。
  - 摘要更新与消息保存共用会话行，存在已知的 `updated_at` 变更影响分页边界。
  - 待确认：是否需要跨会话长期偏好；是否需要向量化记忆；摘要模型是否允许独立配置（当前已支持 `MEMORY_*` 覆盖）。
- **重构适配性**：🟡 部分适合。
- **理由**：LangChain 的 `RunnableWithMessageHistory` / summary memory、LlamaIndex 的 `ChatMemoryBuffer` / `ChatSummaryMemoryBuffer` 可以承接「窗口 + 摘要」的上下文组装和压缩。但本项目的持久化字段、摘要指针、异步触发时机、三处不同裁剪预算、删除消息作废摘要，都是业务语义。建议用框架替换「消息 → 上下文字符串」的组装与摘要器，保留持久化和触发编排。

---

### F12 图片附件问答与视觉理解

- **功能名称**：聊天图片上传、OSS 存储、视觉模型转写、`effective_question` 合并、图片分析状态回传。
- **用途/目标**：让用户可以直接发图提问；把图片内容转成可检索、可生成的文本，同时把识别失败明确告知用户。
- **关键输入与输出**：
  - 输入：PNG/JPEG/WebP 图片（≤5MB）、可选文字问题。
  - 输出：`image_analysis {status, description, error}`；`effective_question`（原文 + 图片描述）。
- **主要流程 / 依赖模块**：
  - `backend/service/chat_service.py::upload_chat_attachment`：类型校验 → 大小校验 → 登记 `chat_attachment_uploads` → OSS PUT（`public-read`）→ 返回 `object_key/url`。
  - `backend/rag/vision_service.py::_build_effective_question` → `_analyze_image_attachments`。
  - 视觉模型走 OpenAI-compatible `chat/completions`，图片以 OSS URL 传入；两个 Prompt 依次重试（先逐字识别，再聚焦细节）；`_classify_image_analysis` 判断 `success/partial/failed`。
  - 只有图片、无文字且识别失败时提前结束，返回 `image_analysis` + `error` SSE 帧。
- **现状边界与待确认点**：
  - 图片存 OSS，且对象 ACL 为 `public-read`（视觉模型需要可访问 URL）；不是签名 URL。
  - 依赖 `VISION_MODEL/VISION_API_KEY`；无本地 OCR、无多图逐张分析、无图片缓存。
  - 图片描述会进入检索和最终 prompt，可能放大上下文长度。
  - 待确认：是否必须公网可读；是否可改为签名 URL/内网代理；是否需要 OCR 兜底；是否支持多图。
- **重构适配性**：🟡 部分适合。
- **理由**：LangChain 的多模态 `HumanMessage`（text + image_url）可以直接替换手写 HTTP 调用，LlamaIndex 也有多模态节点/查询能力。但 OSS 上传、附件登记、状态分类、失败早退和 SSE 帧是业务层。建议只替换视觉模型调用与 Prompt 组装，保留附件生命周期和 `effective_question` 语义。

---

### F13 聊天附件生命周期管理

- **功能名称**：上传登记、发送消费、超时清扫、租约与墓碑、删除消息时的附件回收、`keep_attachments` 跳过墓碑。
- **用途/目标**：解决「上传了但从未发送」的孤儿对象；保证对象归属可追溯；避免删除消息后对象永久泄漏，也避免误删他人对象。
- **关键输入与输出**：
  - 输入：`object_key`、`user_id`、消息附件列表、删除动作。
  - 输出：`chat_attachment_uploads` 行状态迁移（pending → consumed / deleting / reclaimed）；OSS DELETE 外呼结果。
- **主要流程 / 依赖模块**：
  - `backend/crud/chat.py`：`register_attachment_upload`、`confirm_attachment_uploads`、`claim_attachment_upload`、`mark_attachment_reclaimed`、`reclaim_orphan_chat_attachments`。
  - 上传先登记后写对象；发送成功在**同一事务**里消费登记行并写消息；清扫线程按 `CHAT_ATTACHMENT_PENDING_TTL_SECONDS`（默认 24h）领取、写墓碑、租约到期后删 OSS；墓碑保留 7 天。
  - 删除消息时对不再被任何消息引用的键登记墓碑；`keep_attachments=true` 只保留目标消息自身的键。
- **现状边界与待确认点**：
  - 仅支持 OSS；清扫是进程内后台线程，多实例会各自扫描，依赖行级租约避免重复删除。
  - 只为本服务铸造的键签发 DELETE；其他键拒绝，宁可泄漏不误删。
  - 已知边界：`keep_attachments` 删除后重发失败，目标消息附件会停在 consumed 且不再被引用（已知泄漏，代码注释中明确）。
  - 待确认：是否需要换成对象存储生命周期规则；是否需要跨进程任务队列；是否需要附件审计报表。
- **重构适配性**：❌ 不适合。
- **理由**：这是对象存储生命周期与归属状态机，LangChain/LlamaIndex 没有对应抽象。框架的 document store 只管「文档」，不管「上传了但没发」的对象。建议原样保留。

---

### F14 会话/消息管理、导出与反馈

- **功能名称**：会话列表与标题搜索、消息分页、重命名、删除（用户消息截断 / 助手消息单删）、Markdown 导出、消息级赞踩。
- **用途/目标**：提供完整的历史使用闭环；删除消息时联动清理附件、推进会话排序、必要时作废记忆摘要。
- **关键输入与输出**：
  - 输入：`user_id`、`cid`、`message_id`、游标、反馈值 `-1/0/1`。
  - 输出：会话/消息列表；`deleted_message_ids`；Markdown 文件流；`{message_id, feedback}`。
- **主要流程 / 依赖模块**：
  - `backend/service/chat_service.py`：`list_conversations`、`get_messages`、`delete_conversation`、`delete_message`、`rename_conversation`、`submit_message_feedback`。
  - `backend/crud/chat.py::delete_message`：用户消息删除 `id >= 目标` 的后续消息；助手消息只删一条；同一事务内推进 `updated_at`、按需作废记忆摘要、给孤儿附件登记墓碑。
  - `backend/service/chat_export_service.py`：按主键分批读取，输出带来源的 Markdown；文件名做路径/控制字符/Windows 保留名处理。
  - 会话列表按 `(updated_at, id)` 复合游标分页；标题搜索用 `LIKE` 并转义通配符。
- **现状边界与待确认点**：
  - 没有软删除、回收站、会话归档；删除即物理删除。
  - 导出只输出用户可读字段（正文、来源文件名/ID/片段），不导出 Trace 原始 JSON。
  - 反馈只针对 assistant 消息；没有反馈原因标签、没有反馈驱动检索调优。
  - 待确认：是否需要回收站；是否需要导出 JSON/PDF；反馈是否要进入离线评估集。
- **重构适配性**：❌ 不适合。
- **理由**：纯业务 CRUD 与删除语义，与 RAG 框架无关。LangChain/LlamaIndex 的 chat store 不提供「用户消息截断、附件墓碑、记忆摘要作废」这类事务联动。建议保留。

---

### F15 检索 Trace 与可观测性

- **功能名称**：一次问答的逐步学习轨迹（输入、路由、规划、召回、融合、重排、记忆、生成、评估），SSE 实时推送 + 持久化 + 按消息回放。
- **用途/目标**：让用户/运维能看清「为什么是这个答案」；为问题定位、教学演示和后续评估提供结构化数据。
- **关键输入与输出**：
  - 输入：各阶段 `stage/function/creates/uses/params/result/note`。
  - 输出：SSE `trace` 帧；`chat_trace_sessions.events`；assistant 消息的 `retrieval_trace`；`GET /api/chat/traces/{id}` 与 `GET /api/chat/messages/{id}/trace`。
- **主要流程 / 依赖模块**：
  - `backend/rag/learning_trace.py::TraceRecorder`：内存累积事件，每步 `await add` 写回；`drain_sse_payloads` 增量推送给前端；`finish` 写终态。
  - `service/trace_service.py`：SSE 载荷包装、按用户读取、按消息回查。
  - `sanitize_trace_value` / `compact_trace_reference`：截断长文本、压缩引用，避免 Trace 体积失控。
  - `retrieval_trace` 同时落库，包含 embedding 状态、query_plan、routes、rrf、rerank、memory、rag_gate、parent_child、grounding 等。
- **现状边界与待确认点**：
  - 自研事件 schema，前端 `TraceVariableFlow.vue` 和「时间线/变量流/检索过程/记忆管理/RAGAS/原始JSON」标签页依赖它。
  - 错误信息做固定文案收敛，异常原文只进日志；Trace 不含敏感上游地址。
  - 没有 OpenTelemetry / Prometheus 指标 / 分布式链路。
  - 待确认：是否必须保持现有 Trace 事件字段与前端契约；是否要接入标准可观测性后端；Trace 保留时长与体积上限。
- **重构适配性**：🟡 部分适合。
- **理由**：LangChain 的 `BaseCallbackHandler` / `astream_events` 和 LlamaIndex 的 instrumentation 可以在检索、模型调用、重排等节点自动产生事件，减少手工埋点。但本项目的事件 schema、固定文案、SSE 增量推送、前端变量流渲染是契约，必须保留。建议「用框架回调采集底层事件，再映射到现有 Trace schema」，而不是直接暴露框架事件。

---

### F16 RAGAS 在线评估

- **功能名称**：assistant 消息保存后异步评估 faithfulness、context precision（无参考）、response relevancy，并把状态/分数/错误写回消息。
- **用途/目标**：持续观察回答质量，支持按消息查看评估结果；评估失败不影响主回答。
- **关键输入与输出**：
  - 输入：`message_id`、question、answer、retrieved_contexts、trace_id。
  - 输出：`messages.ragas_status`（pending/running/done/failed）、`ragas_scores`、`ragas_error`。
- **主要流程 / 依赖模块**：
  - `backend/rag/ragas_eval.py::schedule_ragas_evaluation` → `evaluate_message_async`。
  - `RAGAS_ENABLED=false` 时只写状态与 Trace，不加载可选依赖。
  - 输入裁剪：最多 3 个 context、每个 1500 字符、回答 2000 字符；总超时 180s，单指标 60s。
  - 指标：`Faithfulness`、`LLMContextPrecisionWithoutReference`、`ResponseRelevancy`；每个指标独立 try，部分成功也写回。
  - 依赖 `requirements-ragas.txt`（`ragas==0.4.3`），未安装时测试显式跳过。
- **现状边界与待确认点**：
  - 默认关闭；需要额外的 LLM + Embedding 配置。
  - 评估在后台 `asyncio.create_task` 中执行，进程重启会丢失未完成任务；没有评估队列/重试。
  - 指标口径固定，未做人工标注校准。
  - 待确认：RAGAS 是否为生产必需；是否要保留三个指标；是否需要离线批量评估（`scripts/eval_rag.py`）与在线评估统一口径。
- **重构适配性**：✅ 适合。
- **理由**：RAGAS 本身已经是独立评估库，且与 LangChain/LlamaIndex 的 LLM/Embedding 适配器兼容。可以把「构造 OpenAI 客户端、llm_factory、embeddings adapter」替换为框架提供的适配器，减少胶水代码。但异步调度、状态机、超时与消息回写要保留。

---

### F17 回答 groundedness 轻量校验

- **功能名称**：对最终回答做逐句词法支撑检查，标记无支撑句子。
- **用途/目标**：在不增加模型调用成本的前提下，给用户/运维一个「回答是否有检索上下文支撑」的快速信号。
- **关键输入与输出**：
  - 输入：answer、retrieved_contexts。
  - 输出：`{status: passed/needs_review/skipped, reason, checked_sentences, unsupported_count, unsupported_claims}`，写入 `retrieval_trace.grounding`。
- **主要流程 / 依赖模块**：
  - `backend/service/utils_service.py::_check_answer_grounding`。
  - 按中英文句末标点/换行切句 → 归一化（保留中文、字母、数字、小数点）→ 数字必须出现在上下文中 → 字符 bigram 重合数 ≥ 6 且重合比例 ≥ 0.18 才判为有支撑。
  - 只记录，不阻断回答，也不修改回答。
- **现状边界与待确认点**：
  - 纯词法启发式，不理解语义；同义改写可能被误判为无支撑。
  - 阈值硬编码，不可配置；最多记录 5 条无支撑句子。
  - 与 RAGAS faithfulness 有功能重叠，但成本与用途不同。
  - 待确认：是否保留这个信号；是否要把它升级为可配置的语义校验；是否要影响前端提示。
- **重构适配性**：❌ 不适合。
- **理由**：LangChain/LlamaIndex 没有等价的确定性词法校验；用 RAGAS faithfulness 替代会引入模型调用、延迟和成本，语义也不同。建议保留为轻量旁路。

---

## P2：工程保障与前端契约

### F18 键集分页与列表查询

- **功能名称**：知识库按 `id` 升序、知识文件按 `id` 降序、会话按 `(updated_at, id)` 复合游标、消息按 `id` 分页。
- **用途/目标**：避免 offset 在数据插入/更新时造成的重复或漏行；让列表接口在数据量增长后仍然稳定。
- **关键输入与输出**：
  - 输入：`limit`、`before_id` / `after_id` / `before_updated_at + before_id`。
  - 输出：一页数据；前端用「多取一条」判断是否还有更多。
- **主要流程 / 依赖模块**：
  - `backend/crud/pagination.py`、`service/pagination_service.py`。
  - `crud/chat.py::list_conversations` 的复合游标展开式，兼容 MySQL/SQLite。
  - 前端 `src/stores/chat.js` 的 `splitPage` / `splitConversationPage` / `conversationCursorOf`。
- **现状边界与待确认点**：
  - 不同资源游标语义不同，新增接口必须沿用同类语义。
  - 会话 `updated_at` 可变，翻页途中被改写的会话可能跳到已翻区间；侧栏每轮问答后重取第一页来收敛。
  - 没有总数统计接口。
  - 待确认：是否需要统一游标封装；是否需要总数；是否需要搜索全文索引。
- **重构适配性**：❌ 不适合。
- **理由**：数据库访问与分页策略，与 LangChain/LlamaIndex 无关。LlamaIndex 的 document store 分页不覆盖本项目的关系库列表语义。

---

### F19 限流、并发闸门与启动守卫

- **功能名称**：登录失败滑动窗口、SSE 聊天并发槽、密钥/数据库口令启动守卫、tiktoken 预热、BM25 预热、孤儿附件清扫循环。
- **用途/目标**：防止暴力登录和资源耗尽；避免用占位密钥/默认口令上线；把重初始化工作移出首请求路径。
- **关键输入与输出**：
  - 输入：环境变量配置（`LOGIN_RATE_LIMIT_*`、`CHAT_STREAM_MAX_CONCURRENCY`、`SECRET_KEY`、`MYSQL_PASSWORD` 等）。
  - 输出：429 + `Retry-After`；启动拒绝；后台预热/清扫任务。
- **主要流程 / 依赖模块**：
  - `backend/service/rate_limit.py`：`FailureWindow`、`ConcurrencyGate`、`Slot`（`release` 幂等）。
  - `backend/main.py::lifespan`：密钥守卫 → tiktoken 预热（10s 超时，非致命）→ `init_db()` → 播种账号 → 可选重建索引 → BM25 预热 → 附件清扫循环。
  - `config.py` 的 `ensure_secret_key_configured` / `ensure_mysql_password_configured`。
- **现状边界与待确认点**：
  - 限流与并发计数在进程内存，多 worker/多实例时上限按进程数放大；没有 Redis。
  - 并发槽不排队，满员立即 429；流交付前后有两段归属，靠 `handed_off` 防止重复释放。
  - 启动重建索引是同步等待（不设总时限），大库会延长启动时间。
  - 待确认：部署形态（单进程/多 worker）、是否需要分布式限流、是否接受启动重建耗时。
- **重构适配性**：❌ 不适合。
- **理由**：进程级工程保障，框架不提供；LangChain/LlamaIndex 的 rate limiter 只针对模型调用，不覆盖 SSE 连接和登录。建议保留。

---

### F20 前端交互与 SSE 协议

- **功能名称**：Vue 3 聊天界面、知识库管理、Markdown 渲染、来源抽屉、Trace 面板、图片分析提示、编辑/重新生成、停止生成、反馈、导出、会话搜索与批量管理。
- **用途/目标**：把后端能力组织成可用的企业问答产品；SSE 协议是前后端共同契约。
- **关键输入与输出**：
  - 输入：SSE `data:` 帧（正文 chunk、`sources`、`conversation`、`image_analysis`、`trace`、`reset`、`error`、`[DONE]`）。
  - 输出：渲染后的消息流、来源卡片、Trace 变量流、知识库文件列表。
- **主要流程 / 依赖模块**：
  - `src/api/chat.js::streamChat` 用原生 `fetch` 读 SSE；`src/utils/streamEvents.js` 解析帧。
  - `CONTROL_EVENT_TYPES = {sources, conversation, image_analysis, trace, reset}`；损坏/截断帧不能降级成正文（issue #134）。
  - `src/stores/chat.js` 管理会话、消息、流式状态、请求序号、分页游标、RAGAS 轮询。
  - `src/views/Chat.vue`、`Knowledge.vue`、`TraceVariableFlow.vue`、`MarkdownRenderer.vue`。
- **现状边界与待确认点**：
  - 前端依赖 Element Plus、Pinia、Vue Router 自动导入；浏览器基线 Safari 16.4+/Chrome 111+/Firefox 128+。
  - SSE 是单向流；没有 WebSocket、没有断线重连续传。
  - 头像需要 token，`<img>` 不能带 Authorization，走 object URL。
  - 待确认：是否允许修改 SSE 帧格式；是否要支持多端/移动端；是否需要断线重连。
- **重构适配性**：❌ 不适合（不在 LangChain/LlamaIndex 范围）。
- **理由**：前端与 SSE 协议不属于检索框架重构；但它是重构时**最强的兼容约束**：任何后端替换都必须继续产出同样的帧类型和字段。建议把 SSE 协议视为冻结契约。

---

## 3. 业务闭环覆盖自检

| 业务闭环 | 覆盖功能 | 是否完整 |
| --- | --- | --- |
| 登录 → 身份 → 权限边界 | F01、F19 | 是 |
| 知识库创建 → 上传 → 解析 → 分块 → 向量化 → 可检索 | F02、F03、F04、F05、F18 | 是 |
| 提问 → 路由 → 规划 → 多路召回 → 融合 → 重排 → 父子聚合 → 生成 → 流式返回 | F06、F07、F08、F09、F10、F20 | 是 |
| 多轮对话 → 短期记忆 → 长期摘要 → 记忆参与检索/生成 | F11 | 是 |
| 图片上传 → 视觉理解 → 合并问题 → 问答 | F12、F13 | 是 |
| 回答落库 → RAGAS 评估 → 质量信号 → 人工反馈 | F14、F16、F17 | 是 |
| 过程可观测 → Trace 回放 → 问题定位 | F15 | 是 |
| 历史管理 → 删除/导出/分页 → 附件回收 | F13、F14、F18 | 是 |
| 安全与稳定性 → 限流/并发/启动守卫 | F01、F19 | 是 |

结论：清单覆盖了「身份与隔离、知识入库、问答主链路、记忆、多模态、质量评估、可观测、历史管理、工程保障」九条主要闭环，没有发现未覆盖的核心业务路径。

---

## 4. 建议的重构分层（仅建议，不执行）

### 4.1 建议保留（不交给框架）

- FastAPI 路由、鉴权、多租户与 `knowledge_base_id` 双隔离。
- 上传/删除的事务编排、失败回滚、入库超时与清理。
- BM25 的 SQL 预筛、分批游标、语料缓存/预热、Unicode 归一化。
- `select_final_chunks` 的关键词补位、去重键与最终 top-N。
- SSE 帧协议、`reset` 语义、Trace 事件 schema、失败文案收敛。
- 附件登记/消费/租约/墓碑/清扫状态机。
- 键集分页、限流、并发闸门、启动守卫。
- 确定性 groundedness 校验。

### 4.2 建议替换为 LangChain / LlamaIndex

| 能力 | 建议承接方 | 替换边界 |
| --- | --- | --- |
| 文档读取 | LlamaIndex `SimpleDirectoryReader` / `DocxReader` / `PDFReader` | 只替换抽取器，保留上传事务与空文本拒绝。 |
| Node 切分与节点关系 | LlamaIndex `NodeParser` / `HierarchicalNodeParser` | 保留中文标题/条款/表格规则与 token 预算。 |
| Embedding | LangChain `Embeddings` 适配器 | 保留批处理、429/5xx 重试、deadline、失败不降级。 |
| 向量存储 | LlamaIndex `MilvusVectorStore` | 保留 `knowledge_base_id` 过滤与替换语义。 |
| 多路/混合召回 | LlamaIndex `QueryFusionRetriever`、`VectorIndexRetriever`、`BM25Retriever` | 向量多路可替换；关键词 SQL 预筛保留为自定义 Retriever。 |
| RRF 融合 | LlamaIndex `QueryFusionRetriever` 的 RRF 模式 | 保留 `chunk_key` 去重与关键词补位。 |
| Rerank | LlamaIndex reranker 模块 / LangChain `ContextualCompressionRetriever` | 保留 DashScope 主服务与 LLM 兜底语义。 |
| 父子聚合 | LlamaIndex `AutoMergingRetriever` / `ParentDocumentRetriever` | 保留 MySQL 元数据表与 token 预算裁剪。 |
| 查询规划 | LangChain structured output / LlamaIndex `HyDEQueryTransform`、`SubQuestionQueryEngine` | 保留低置信度保守进入 RAG 与确定性兜底。 |
| Memory | LangChain memory / LlamaIndex `ChatMemoryBuffer` + `ChatSummaryMemoryBuffer` | 保留会话行持久化、摘要指针、异步触发。 |
| 多模态 | LangChain `HumanMessage`（text + image_url） | 保留 OSS 附件生命周期与图片分析状态。 |
| RAGAS | RAGAS + 框架 LLM/Embedding 适配器 | 保留异步调度、状态机、超时与消息回写。 |
| 可观测性 | LangChain callbacks / LlamaIndex instrumentation | 映射到现有 Trace schema，不直接暴露框架事件。 |

### 4.3 建议的迁移顺序

1. **先换 Embedding + 向量存储适配层**（F05）：影响面最小，可用现有索引做回归对比。
2. **再换文档解析 + Node 切分**（F03/F04）：用离线语料对比切片数量、覆盖率、检索命中。
3. **再换检索组合与 rerank**（F07/F08/F09）：保留关键词路与最终选择，逐步替换向量多路和融合。
4. **最后换 Memory / 多模态 / RAGAS 接入**（F11/F12/F16）：这些是旁路，失败不影响主回答。
5. **始终不动**：F01/F02/F13/F14/F17/F18/F19/F20。

---

## 5. 待确认清单（重构前必须回答）

1. **部署形态**：单进程 uvicorn 还是多 worker/多实例？这直接影响限流、BM25 缓存、Milvus Lite 单例、附件清扫的语义。
2. **数据规模**：知识库数量、文件数量、单文件最大字符数、总窗口数、日活/并发问答量。决定 BM25 上限、父子切片开关、向量库选型。
3. **索引迁移窗口**：是否允许停机重建索引？是否允许修改 Milvus collection schema？是否需要双写/灰度？
4. **契约冻结范围**：SSE 帧类型与字段、Trace 事件 schema、`sources` 字段、`retrieval_trace` 结构，前端与 E2E 是否要求逐字段兼容？
5. **失败降级语义**：是否必须保留「路由失败保守进入 RAG」「Embedding 不可用只跳过向量路」「Rerank 失败回退 LLM / 回退 fused」「生成失败不落库」？
6. **数据事实来源**：知识文件正文、切片元数据是否必须继续以 MySQL 为唯一事实来源？是否允许迁到对象存储/文档库？
7. **多租户模型**：是否只需要「用户私有知识库」，还是需要组织/部门共享、角色权限、审计日志？
8. **图片存储安全**：OSS `public-read` 是否可改为签名 URL 或内网代理？视觉模型是否必须公网访问？
9. **RAGAS 定位**：生产必需还是可选观测？三个指标是否保留？是否需要离线评估集与在线评估统一口径？
10. **中文检索口径**：BM25 是否需要与现有实现逐条等价？是否允许引入词典分词或替换为框架 BM25？
11. **Memory 范围**：是否只需要会话内记忆，还是需要跨会话用户偏好/画像？
12. **可观测性**：是否要接入 OpenTelemetry/Prometheus？Trace 保留时长、体积上限、脱敏要求是什么？
13. **测试基线**：是否以现有 `tests/` 与 `tests/fixtures/rag_eval_questions.json` 作为重构回归基线？是否接受检索结果排序变化？
14. **框架版本与许可**：LlamaIndex 的引入版本、是否允许其默认 telemetry、依赖体积与 Python 3.10 兼容性。

---

## 6. 附：主要证据索引

| 主题 | 代码位置 |
| --- | --- |
| 聊天主链路 | `backend/service/chat_service.py::stream_chat` |
| 检索主链路 | `backend/rag/retrieval.py::retrieve_knowledge` |
| RAG 路由 / 查询规划 | `backend/rag/retrieval.py::decide_need_rag`、`build_query_plan` |
| RRF | `backend/rag/retrieval.py::rrf_fuse` |
| 父子聚合 | `backend/rag/retrieval.py::expand_parent_chunks` |
| 关键词/BM25 | `backend/rag/retrieval.py::keyword_recall`、`backend/rag/bm25.py` |
| 重排与最终选择 | `backend/rag/rerank.py::rerank_chunks`、`select_final_chunks` |
| Embedding / Milvus | `backend/rag/milvus_client.py` |
| 分块与父子切片 | `backend/crud/knowledge_file.py::chunk_text`、`build_parent_child_chunks` |
| 知识上传 | `backend/service/knowledge_service.py::upload_knowledge` |
| 会话记忆 | `backend/rag/memory_service.py` |
| 流式生成 / 后备模型 | `backend/rag/llm.py::stream_answer_events` |
| 视觉理解 | `backend/rag/vision_service.py` |
| RAGAS | `backend/rag/ragas_eval.py` |
| Trace | `backend/rag/learning_trace.py::TraceRecorder` |
| 附件生命周期 | `backend/crud/chat.py`、`backend/service/chat_service.py::reclaim_orphan_chat_attachments` |
| 数据模型 | `backend/model/models.py` |
| 配置 | `backend/config.py` |
| SSE 解析 | `src/utils/streamEvents.js` |
| 聊天状态 | `src/stores/chat.js` |
| 前端页面 | `src/views/Chat.vue`、`src/views/Knowledge.vue` |

---

> 本文的适配性判断基于当前代码与配置，不构成对 LangChain/LlamaIndex 具体版本的兼容性承诺；实施前应以第 5 节待确认项和现有测试基线为准。
