# 后端 RAG 实现分析

> 分析范围：`backend/` 下的后端代码、依赖声明与配置。
> 结论以仓库当前文件为准；行号对应分析时的文件内容。

## 1. 结论

**本项目没有使用某个完整的 RAG 框架（如 LlamaIndex、Haystack、LangChain RetrievalQA 等）来实现 RAG。RAG 主链路是自研编排代码，基础库的组合如下：**

| 能力 | 使用的库/组件 | 说明 |
| --- | --- | --- |
| LLM 对话与流式生成 | `langchain-openai` 的 `ChatOpenAI` | 只用它封装 OpenAI-compatible Chat Completions；路由判断、查询规划、回答生成、记忆摘要、重排 LLM 兜底都通过它调用 |
| 向量数据库 | `pymilvus` + `milvus-lite` | 本地 Milvus Lite 文件或远程 Milvus；自定义 collection schema，`AUTOINDEX` + `COSINE` |
| Embedding | 自研 HTTP 客户端（`httpx`） | 调用 OpenAI-compatible `/embeddings` 接口，默认 DashScope `text-embedding-v4`，维度 1024；不是 LangChain Embeddings |
| Rerank | 自研 HTTP 客户端（`httpx`） | 主链路调用 DashScope `qwen3-rerank`；失败时用 `ChatOpenAI` 做 JSON 重排兜底 |
| 关键词召回 | 自研 BM25 | `backend/rag/bm25.py`，没有使用 `rank_bm25` 等第三方 BM25 库 |
| 融合 | 自研 RRF | `backend/rag/retrieval.py::rrf_fuse()` |
| 查询规划 / HyDE / 改写 | 自研 Prompt + JSON 解析 | `decide_need_rag()`、`build_query_plan()`，不是 LangChain Retriever/MultiQueryRetriever |
| 父子切片 | 自研 | `_chunk_for_index()`、`expand_parent_chunks()`，不是 LangChain ParentDocumentRetriever |
| 在线评估 | RAGAS（可选） | `ragas==0.4.3`，只在 assistant 消息落库后异步评估；默认关闭，懒加载 |

一句话概括：**LangChain 只作为“OpenAI-compatible 聊天模型客户端”使用；检索、召回、融合、重排、切片、记忆和评估编排全部由仓库自己的代码完成。**

## 2. 证据

### 2.1 依赖声明与版本

#### 主运行依赖

`backend/requirements.txt` 是 CI 与运行时安装的依赖文件（`docs/TESTING.md` 也以它为准）：

```text
11  httpx==0.28.1
14  langchain-openai==1.6.7
15  openai>=3.23.0
16  pymilvus[milvus_lite]==3.0.2
17  milvus-lite==3.2.1
19  tiktoken==0.14.0
```

- `langchain-openai==1.6.7`：`backend/requirements.txt:14`。
- `openai>=3.23.0`：`backend/requirements.txt:15`，主要用于异常类型和 RAGAS 的 OpenAI 客户端。
- `pymilvus[milvus_lite]==3.0.2`：`backend/requirements.txt:16`。
- `milvus-lite==3.2.1`：`backend/requirements.txt:17`。
- `tiktoken==0.14.0`：`backend/requirements.txt:19`，用于 token 计数与分块预算，不是 embedding 库。

`backend/pyproject.toml`（当前为未跟踪文件）声明了相同的关键版本：

```text
11  "langchain-openai==1.6.7",
12  "milvus-lite==3.2.1",
14  "openai>=3.23.0",
16  "pymilvus[milvus-lite]==3.0.2",
24  "tiktoken==0.14.0",
```

#### 实际解析/安装版本

仓库根目录的 `uv.lock`（当前为未跟踪文件）解析出的版本：

- `langchain-core==1.7.4`：`uv.lock:873-874`
- `langchain-openai==1.6.7`：`uv.lock:895-896`
- `milvus-lite==3.2.1`：`uv.lock:1057-1058`
- `openai==3.45.0`：`uv.lock:1145-1146`
- `pymilvus==3.0.2`：`uv.lock:1511-1512`
- `tiktoken==0.14.0`：`uv.lock:1884-1885`

当前根虚拟环境中实际安装版本也一致：`langchain-openai 1.6.7`、`langchain-core 1.7.4`、`openai 3.45.0`、`pymilvus 3.0.2`、`milvus-lite 3.2.1`、`tiktoken 0.14.0`；`ragas` 未安装。

#### 可选的 RAGAS 依赖

`backend/requirements-ragas.txt`：

```text
16  ragas==0.4.3
26  langchain-community==0.3.30
```

该文件只用于可选的在线评估（`RAGAS_ENABLED` 默认 `False`），不属于 RAG 主链路：

- `ragas==0.4.3`：`backend/requirements-ragas.txt:16`
- `langchain-community==0.3.30`：`backend/requirements-ragas.txt:26`，文件注释说明它是为了兼容 RAGAS 0.4.x 对 `langchain_community.chat_models.vertexai` 的导入，不是主应用使用的 LangChain RAG 组件。

### 2.2 LangChain 的实际使用范围

主链路只有一处 LangChain 导入：

```python
# backend/rag/llm.py:7
from langchain_openai import ChatOpenAI
```

`backend/rag/llm.py:8-16` 另外从 `openai` 导入异常类型（`APIConnectionError`、`APIStatusError`、`RateLimitError` 等），用于分类生成失败。

`ChatOpenAI` 的构造位置：

- `get_deepseek_model()`：`backend/rag/llm.py:175-189`，把 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 传进 `ChatOpenAI`。
- `get_text_fallback_model()`：`backend/rag/llm.py:192-206`。
- `get_stage_model()`：`backend/rag/llm.py:287-339`，按 RAG 阶段选择模型；阶段与继承关系在 `backend/rag/llm.py:212-229`。

调用方式：

- 非流式文本：`call_chat_text_with_metadata()` 使用 `model.ainvoke(...)`，`backend/rag/llm.py:342-357`。
- JSON 结构化调用：`call_chat_json()`、`call_router_json()`，`backend/rag/llm.py:373-412`。
- 流式生成：`stream_answer_events()`，`backend/rag/llm.py:415-553`；实际流式读取在 `_stream_model_chunks()`，`backend/rag/llm.py:582-585`。
- `backend/rag/chains.py:1-13` 的 `stream_rag_answer()` 只是 `stream_answer_events()` 的薄包装，不是 LangChain Chain。

**没有发现**以下 LangChain RAG 组件的导入或使用：

- `langchain.chains.RetrievalQA`
- `VectorStoreRetriever` / `Retriever`
- `langchain_text_splitters` / `TextSplitter`
- `DocumentLoader`
- LangChain 的向量库集成（如 `langchain_community.vectorstores.*`）
- `MultiQueryRetriever` / `ParentDocumentRetriever` / `HypotheticalDocumentEmbedder`

证据：对 `backend/` 的静态导入扫描只命中 `backend/rag/llm.py:7` 的 `from langchain_openai import ChatOpenAI`；`Chain`、`Retriever`、`VectorStore`、`TextSplitter` 等名称在 `.py` 文件中没有出现。

### 2.3 Milvus 向量库

`backend/rag/milvus_client.py` 直接使用 `pymilvus`：

```python
# backend/rag/milvus_client.py:64-65
from pymilvus import DataType, MilvusClient  # noqa: E402
from pymilvus.orm.schema import CollectionSchema, FieldSchema  # noqa: E402
```

- collection 名称：`_collection_name()`，`backend/rag/milvus_client.py:75-82`；默认由 embedding 模型名和维度拼成。
- collection schema：`_collection_schema()`，`backend/rag/milvus_client.py:424-434`，字段为：
  `id`（VARCHAR 主键）、`embedding`（FLOAT_VECTOR，维度 `EMBEDDING_DIM`）、`content`（VARCHAR）、`file_id`（INT64）、`file_name`（VARCHAR）、`chunk_id`（VARCHAR）、`knowledge_base_id`（INT64）。
- 索引：`_index_params()`，`backend/rag/milvus_client.py:437-440`，使用 `AUTOINDEX` + `COSINE`。
- 连接与建 collection：`_connect_milvus()`、`_ensure_collection()`，`backend/rag/milvus_client.py:408-452`；本地 Milvus Lite 使用单例和文件锁，`backend/rag/milvus_client.py:410-421`。
- 查询：`query_vectors()`，`backend/rag/milvus_client.py:724-765`；先对 query 做 embedding，再调用 `client.search(...)`，并用 `knowledge_base_id == ...` 过滤，`backend/rag/milvus_client.py:737-746`。
- 写入：`add_chunks()`，`backend/rag/milvus_client.py:674-702`；先向量化，再删除同一 `file_id` 的旧向量，最后 `client.insert(...)`。
- 删除：`delete_file_chunks()`，`backend/rag/milvus_client.py:718-721`。

Milvus 连接参数来自 `backend/config.py:139-145`：

- `MILVUS_LITE_URI`（默认 `./milvus.db`）
- `MILVUS_URI`（默认继承 `MILVUS_LITE_URI`）
- `MILVUS_TOKEN`、`MILVUS_USER`、`MILVUS_PASSWORD`、`MILVUS_DB_NAME`、`MILVUS_COLLECTION_NAME`

### 2.4 Embedding：自研 HTTP 客户端，不是 LangChain Embeddings

`backend/rag/milvus_client.py` 的 `_OpenAICompatibleEmbeddingFunction` 直接向 OpenAI-compatible `/embeddings` 接口发请求：

```python
# backend/rag/milvus_client.py:332-385
class _OpenAICompatibleEmbeddingFunction:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        ...
        payload = {
            "model": EMBEDDING_MODEL,
            "input": list(texts),
            "dimensions": EMBEDDING_DIM,
        }
        ...
        with httpx.Client(timeout=_embedding_timeout_seconds()) as client:
            response = client.post(url, json=payload, headers=headers)
```

关键位置：

- embedding URL：`_embeddings_url()`，`backend/rag/milvus_client.py:120-122`。
- 配置：`backend/config.py:252-258`，默认 `EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`、`EMBEDDING_MODEL=text-embedding-v4`、`EMBEDDING_DIM=1024`；`EMBEDDING_API_KEY` 为空时回退到 `DASHSCOPE_API_KEY`。
- 批量入库与重试：`_embed_documents()`，`backend/rag/milvus_client.py:583-671`；批大小和 token 预算由 `EMBEDDING_MAX_BATCH_ITEMS`、`EMBEDDING_MAX_REQUEST_TOKENS` 等控制，`backend/config.py:273-295`。
- 开发期兜底：当 `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` 都没配置时，`_OpenAICompatibleEmbeddingFunction` 会使用 `_HashEmbeddingFunction` 生成随机哈希向量（`backend/rag/milvus_client.py:334-338`、`backend/rag/milvus_client.py:100-113`）。这是本地开发兜底，不是语义 embedding；检索链路在 embedding 调用失败时也不会回退到哈希向量，而是跳过该向量路并继续关键词路（`backend/rag/retrieval.py:294-304`）。

### 2.5 Rerank：DashScope HTTP + LLM 兜底

`backend/rag/rerank.py` 没有使用第三方 rerank 框架：

- 入口：`rerank_chunks()`，`backend/rag/rerank.py:31-60`；先尝试 `qwen3-rerank`，失败后按配置切 LLM 兜底。
- 主重排：`_rerank_chunks_with_qwen3()`，`backend/rag/rerank.py:63-112`，用 `httpx.AsyncClient` POST 到 `RERANK_BASE_URL`，请求体包含 `model`、`query`、`documents`、`top_n`、`instruct`，`backend/rag/rerank.py:67-80`。
- LLM 兜底：`_rerank_chunks_with_llm()`，`backend/rag/rerank.py:151-203`，通过 `call_chat_json(..., stage="rerank_llm")` 让模型返回 JSON 重排结果。
- 最终选择：`select_final_chunks()`，`backend/rag/rerank.py:206-219`；在 rerank 结果基础上补充命中了查询关键词的最佳关键词候选。

配置在 `backend/config.py:405-413`：`RERANK_PROVIDER`、`RERANK_MODEL`、`RERANK_BASE_URL`、`RERANK_API_KEY`、`RERANK_TIMEOUT_SECONDS`、`RERANK_LLM_FALLBACK_ENABLED`。

### 2.6 BM25 与 RRF：自研实现

BM25 在 `backend/rag/bm25.py` 中从零实现：

- 模块说明：`backend/rag/bm25.py:1-32`。
- 分词：`tokenize()`，`backend/rag/bm25.py:96-113`，包含 NFKC 归一化、中文 unigram/bigram、ASCII 词和错误码/条款号处理。
- 语料统计：`BM25Index`，`backend/rag/bm25.py:121-215`。
- 没有 `rank_bm25` 或类似的第三方依赖。

检索侧使用位置：

- 关键词召回入口：`keyword_recall()`，`backend/rag/retrieval.py:441`。
- 排序实现：`_keyword_recall_ranked()`，`backend/rag/retrieval.py:579`。
- BM25 索引构建与打分：`_build_bm25_index()`，`backend/rag/retrieval.py:948`；`_score_bm25()`，`backend/rag/retrieval.py:1018`。
- 配置：`backend/config.py:464-479`（`BM25_ENABLED`、`BM25_K1`、`BM25_B`、`BM25_MAX_DOCS`、`BM25_MAX_TERMS`、`BM25_MAX_SCORED_CANDIDATES`）。

RRF 融合：

```python
# backend/rag/retrieval.py:1353
def rrf_fuse(route_results: list[tuple[str, list[dict]]], k: int = 60) -> list[dict]:
```

`retrieve_knowledge()` 在 `backend/rag/retrieval.py:327` 调用 `rrf_fuse()`，随后 `rerank_chunks()`（`:330`）、`select_final_chunks()`（`:332`）、可选的父子切片展开（`:336`）。

### 2.7 查询规划、路由与生成

- RAG 路由：`decide_need_rag()`，`backend/rag/retrieval.py:162-181`；通过 `call_router_json()` 调用 `ChatOpenAI`，失败时保守地进入 RAG。
- 查询规划：`build_query_plan()`，`backend/rag/retrieval.py:184-212`；生成 `hyde_document`、`rewrites`、`keywords`，通过 `call_chat_json()` 调用模型。
- 多路 query 构造：`_build_route_specs()`，`backend/rag/retrieval.py:383-395`；包含 `planned`、`simplified`、`sub_question_N`、`hyde`、`rewrite_N`。
- 检索主流程：`retrieve_knowledge()`，`backend/rag/retrieval.py:215-353`。
  - 向量路与关键词路并发：`asyncio.gather(..., return_exceptions=True)`，`backend/rag/retrieval.py:290`。
  - 向量路：`query_vectors`，`backend/rag/retrieval.py:262`。
  - 关键词路：`_keyword_recall_step` / `keyword_recall`，`backend/rag/retrieval.py:277`、`:441`。
  - embedding 失败只跳过该路，关键词路继续：`backend/rag/retrieval.py:294-304`。
- 回答生成：`stream_answer_events()`，`backend/rag/llm.py:415-553`；Prompt 构造在 `build_answer_messages()`，`backend/rag/llm.py:554-581`；流式读取在 `_stream_model_chunks()`，`backend/rag/llm.py:582-585`。
- 聊天服务编排：`stream_chat()`，`backend/service/chat_service.py:783`。
  - 构造 effective question：`backend/service/chat_service.py:847`（`_build_effective_question`）。
  - 构造记忆上下文与检索问题：`backend/service/chat_service.py:990-994`。
  - RAG 路由判断：`backend/service/chat_service.py:1016`。
  - 调用 `retrieve_knowledge()`：`backend/service/chat_service.py:1074`。
  - 拼接知识库上下文与 sources：`backend/service/chat_service.py:1112-1118`。
  - 调用 `stream_rag_answer()`：`backend/service/chat_service.py:1192`。
  - 异步 RAGAS：`backend/service/chat_service.py:1322`。

### 2.8 会话记忆与图片理解

这些是 RAG 链路的辅助能力，也都是自研实现：

- 记忆上下文：`backend/rag/memory_service.py:80-106`；短期窗口读取 `:33-79`，摘要更新 `:157-229`，摘要压缩 `:349-406`。
- 图片理解：`_build_effective_question()`，`backend/rag/vision_service.py:21`；实际请求图片描述 `_request_image_description()`，`backend/rag/vision_service.py:129-155`，用 `httpx.AsyncClient` 调用 OpenAI-compatible Chat Completions，把图片描述与用户文本合并成 `effective_question`。
- 视觉能力不是 RAG 框架的一部分，只影响检索问题的构造。

### 2.9 RAGAS：仅用于可选的在线评估

`backend/rag/ragas_eval.py` 在函数内部延迟导入 RAGAS，主链路不会在未启用时加载：

```python
# backend/rag/ragas_eval.py:182-186
from openai import OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
```

- 开关：`RAGAS_ENABLED`，`backend/config.py:416`，默认 `False`。
- 调度：`schedule_ragas_evaluation()`，`backend/rag/ragas_eval.py:148-166`。
- 评估入口：`evaluate_message_async()`，`backend/rag/ragas_eval.py:38-146`；同步实现 `_evaluate_message_sync()`，`backend/rag/ragas_eval.py:176-208`。
- 指标：`Faithfulness`、`LLMContextPrecisionWithoutReference`、`ResponseRelevancy`，`backend/rag/ragas_eval.py:204-208`。
- 依赖：`backend/requirements-ragas.txt:16` 的 `ragas==0.4.3`；CI 默认不安装。

RAGAS 只读取已生成的回答和检索到的上下文，不参与检索、融合、重排或回答生成。

### 2.10 未发现的 RAG 框架

静态扫描 `backend/` 的依赖声明与 `.py` 导入后，**没有发现**以下框架/库的使用：

- LlamaIndex（`llama_index`）
- Haystack（`haystack`）
- Chroma（`chromadb`）
- FAISS（`faiss`）
- Qdrant（`qdrant_client`）
- Weaviate（`weaviate`）
- LangGraph（`langgraph`）
- LangChain 的 `RetrievalQA`、`VectorStoreRetriever`、`TextSplitter`、`DocumentLoader`
- `rank_bm25` / `sentence-transformers` / `unstructured`

实际命中的 RAG 相关导入只有：

```text
backend/rag/llm.py:7              from langchain_openai import ChatOpenAI
backend/rag/llm.py:8-16           from openai import (...异常类型...)
backend/rag/milvus_client.py:64-65 from pymilvus import ...
backend/rag/ragas_eval.py:182-186  from ragas... / from openai import OpenAI
```

`backend/rag/vision_service.py:10` 的 `from rag.llm import openai_chat_url` 是仓库内的 URL 辅助函数，不是外部 OpenAI RAG 框架。

## 3. 实现流程

### 3.1 入库流程

```text
POST /api/knowledge/upload
  -> backend/service/knowledge_service.py::upload_knowledge()        (:551)
  -> 文件类型/大小校验                                               (:554-561)
  -> crud.knowledge_file.extract_file_text() 抽取文本                (:567; crud/knowledge_file.py:162)
  -> _chunk_for_index() 切片                                         (:600-602; knowledge_service.py:316-328)
       -> crud.knowledge_file.chunk_text()                          (crud/knowledge_file.py:817)
       -> 可选父子切片 build_parent_child_chunks()                   (knowledge_service.py:325-327)
  -> crud.knowledge_file.replace_chunk_metadata() 写 MySQL 元数据    (:616; crud/knowledge_file.py:1683)
  -> rag.milvus_client.add_chunks()                                  (:637-644; milvus_client.py:674-702)
       -> _embed_documents() 分批向量化                              (milvus_client.py:583-671)
       -> _OpenAICompatibleEmbeddingFunction 调 embedding HTTP       (milvus_client.py:332-385)
       -> Milvus client.insert()                                     (milvus_client.py:702)
  -> 失败时 delete_file_chunks() 清理已写向量                        (:660; milvus_client.py:718-721)
```

关键点：

- 切片默认按 token 预算（`CHUNK_TOKEN_LIMIT/TARGET/OVERLAP`），不是固定字符窗口；`chunk_text()` 的两种模式说明见 `backend/crud/knowledge_file.py:817-833`。
- 向量写入采用“先算新向量，再删旧向量，再插入”的替换语义，见 `backend/rag/milvus_client.py:686-702`。
- 切片结构元数据写 MySQL 的 `knowledge_chunk_metadata`，Milvus schema 不增加这些字段，见 `backend/model/models.py:138-173`。

### 3.2 查询流程

```text
POST /api/chat/stream
  -> backend/service/chat_service.py::stream_chat()                  (:783)
  -> _build_effective_question()：文本 + 图片描述                    (:847; vision_service.py:21)
  -> _build_memory_context() / _build_memory_aware_retrieval_question() (:990-994)
  -> decide_need_rag()                                               (:1016; retrieval.py:162)
       -> call_router_json() -> ChatOpenAI.ainvoke()                 (retrieval.py:176; llm.py:391-412)
  -> need_rag=True 时：
       retrieve_knowledge()                                         (:1074; retrieval.py:215-353)
         -> build_query_plan()                                      (retrieval.py:184-212)
         -> _build_route_specs()                                    (retrieval.py:383-395)
         -> 并行召回：
              query_vectors() -> Milvus + embedding                 (retrieval.py:262; milvus_client.py:724-765)
              keyword_recall() -> 自研 BM25                         (retrieval.py:277,441; bm25.py)
         -> rrf_fuse()                                              (retrieval.py:1353)
         -> rerank_chunks()                                         (retrieval.py:330; rerank.py:31-60)
              -> DashScope qwen3-rerank HTTP                        (rerank.py:63-112)
              -> 失败时 LLM JSON 重排兜底                            (rerank.py:151-203)
         -> select_final_chunks()                                   (retrieval.py:332; rerank.py:206-219)
         -> expand_parent_chunks()（可选）                          (retrieval.py:336,1534)
  -> 拼接 context / sources                                         (:1112-1118)
  -> stream_rag_answer() -> stream_answer_events() -> ChatOpenAI.astream() (:1192; chains.py:4-13; llm.py:415-553)
  -> SSE 返回文本 chunk / sources / trace / error / [DONE]
```

### 3.3 生成后评估

```text
assistant 消息落库
  -> schedule_ragas_evaluation()                                    (chat_service.py:1322; ragas_eval.py:148)
  -> evaluate_message_async()                                      (ragas_eval.py:38)
  -> RAGAS_ENABLED=false：只写状态和 Trace，不加载 ragas            (ragas_eval.py:45-55)
  -> RAGAS_ENABLED=true：延迟导入 ragas，调用三个指标                (ragas_eval.py:176-208)
```

RAGAS 是“评估”而不是“RAG 实现”：它消费已经生成的 answer 和 contexts，不改变检索或生成结果。

## 4. 配置与依赖

### 4.1 配置分组

| 配置组 | 变量 | 默认/说明 | 定义位置 | 主要使用位置 |
| --- | --- | --- | --- | --- |
| LLM 主模型 | `DEEPSEEK_*` | DeepSeek OpenAI-compatible；回答、规划、记忆、重排兜底 | `config.py:158-160` | `rag/llm.py:175-189` |
| LLM 文本兜底 | `TEXT_FALLBACK_*` | DashScope/OpenAI-compatible 文本兜底 | `config.py:192-198` | `rag/llm.py:192-206` |
| LLM 阶段覆盖 | `ROUTER_*`、`PLANNER_*`、`ANSWER_*`、`MEMORY_*`、`RERANK_LLM_*`、`RAGAS_*` | 留空继承；router 继承 `TEXT_FALLBACK_*`，其余继承 `DEEPSEEK_*` | `config.py:165-179`、`:202-204`、`:418-420` | `rag/llm.py:212-229`、`:287-339` |
| Milvus | `MILVUS_LITE_URI`、`MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_USER`、`MILVUS_PASSWORD`、`MILVUS_DB_NAME`、`MILVUS_COLLECTION_NAME` | 本地 Lite 文件或远程 Milvus | `config.py:139-145` | `rag/milvus_client.py:392-452` |
| Embedding | `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_TIMEOUT_SECONDS`、`EMBEDDING_MAX_*`、`EMBEDDING_INGEST_*`、`TIKTOKEN_CACHE_DIR` | 默认 DashScope `text-embedding-v4`，1024 维；单条 8192 token、单批 10 条 | `config.py:252-295` | `rag/milvus_client.py:332-385`、`:583-671` |
| 分块 | `CHUNK_TOKEN_LIMIT`、`CHUNK_TOKEN_TARGET`、`CHUNK_TOKEN_OVERLAP`、`CHUNK_TOKENIZER`、`CHUNK_TOKEN_SAFETY_RATIO` | 默认 500/400/50 tokens，`cl100k_base`，安全系数 0.85 | `config.py:305-366` | `crud/knowledge_file.py:817+`、`rag/milvus_client.py`（token 估算） |
| 父子切片 | `PARENT_CHILD_ENABLED`、`PARENT_CONTEXT_MAX_TOKENS` | 默认关闭；父级上下文上限 6000 tokens | `config.py:375-381` | `service/knowledge_service.py:316-328`、`rag/retrieval.py:1390-1599` |
| 入库预算 | `KNOWLEDGE_UPLOAD_MAX_MB`、`KNOWLEDGE_INDEX_MAX_WORKERS`、`KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS`、`EMBEDDING_INGEST_TIMEOUT_SECONDS` | 上传大小、线程池、整份文档预算 | `config.py:155`、`:388-402` | `service/knowledge_service.py:551-680` |
| 检索 | `RETRIEVAL_ROUTE_TOP_K`、`RETRIEVAL_RERANK_TOP_N` | 每路 top-k 8，最终 top-n 5 | `config.py:428-429` | `rag/retrieval.py:215+`、`rag/rerank.py:72` |
| 重排 | `RERANK_PROVIDER`、`RERANK_MODEL`、`RERANK_BASE_URL`、`RERANK_API_KEY`、`RERANK_TIMEOUT_SECONDS`、`RERANK_LLM_FALLBACK_ENABLED` | 默认 DashScope `qwen3-rerank`，失败可切 LLM | `config.py:405-413` | `rag/rerank.py:10-18`、`:63-203` |
| BM25 | `BM25_ENABLED`、`BM25_K1`、`BM25_B`、`BM25_MAX_DOCS`、`BM25_MAX_TERMS`、`BM25_MAX_SCORED_CANDIDATES` | 默认开启；分别限制语料窗口、词表和完整打分候选数 | `config.py:464-479` | `rag/retrieval.py:32-37`、`:441`、`:579`、`:948`、`:1018` |
| RAGAS | `RAGAS_ENABLED`、`RAGAS_BASE_URL`、`RAGAS_API_KEY`、`RAGAS_LLM_MODEL`、`RAGAS_*_TIMEOUT_SECONDS`、`RAGAS_MAX_*` | 默认关闭；启用前需安装可选依赖 | `config.py:416-425` | `rag/ragas_eval.py:38-208` |
| 记忆 | `MEMORY_WINDOW_TURNS`、`MEMORY_SUMMARY_MAX_TOKENS`、`MEMORY_RECENT_MAX_TOKENS`、`MEMORY_*_MAX_CHARS` | 短期窗口轮数与摘要预算 | `config.py:482+` | `rag/memory_service.py` |
| 学习 Trace | `LEARNING_TRACE_ENABLED`、`LEARNING_TRACE_MAX_TEXT_CHARS` | 默认开启 | `config.py:491+` | `rag/learning_trace.py` |

`.env.example` 是配置模板，相关行段：

- Milvus：`.env.example:18-27`
- DeepSeek：`.env.example:44-46`
- DashScope 公共密钥：`.env.example:51`
- 文本兜底：`.env.example:60-63`
- 阶段覆盖：`.env.example:69-85`
- Embedding：`.env.example:88-116`
- 分块：`.env.example:130-134`
- 父子切片：`.env.example:141-145`
- Rerank：`.env.example:148-153`
- RAGAS：`.env.example:156-162`
- Retrieval：`.env.example:165-166`
- BM25：`.env.example:188-193`

### 4.2 依赖关系总结

- **生成模型**：`langchain-openai.ChatOpenAI`，底层通过 `openai` SDK 访问 OpenAI-compatible API；DeepSeek 与 DashScope 都通过 `base_url` 切换。
- **向量化**：`httpx` 直连 OpenAI-compatible `/embeddings`；没有使用 LangChain Embeddings。
- **向量存储**：`pymilvus` 客户端 + `milvus-lite` 本地实现；没有使用 Chroma/FAISS/Qdrant/Weaviate。
- **关键词排序**：纯 Python 自研 BM25；没有 `rank_bm25`。
- **评估**：可选 `ragas`；它的 `langchain-community` 依赖只是兼容性 pin，不参与主链路。
- **文本处理**：`tiktoken` 做 token 估算；`python-docx`、`pypdf` 做文档解析；这些都不是 RAG 框架。

## 5. 未确认项与假设

1. **版本漂移**：`backend/requirements.txt` 是 CI/运行时的事实来源；`backend/pyproject.toml` 与仓库根目录的 `uv.lock` 当前是未跟踪文件，且 `pyproject.toml:5` 要求 `requires-python = ">=3.12"`，与 CI/`runtime.txt` 使用的 Python 3.10 不一致。若重新解析依赖，`langchain-core` 等传递依赖版本可能变化；本文的版本结论以当前 `requirements.txt`、`uv.lock` 和已安装环境为准。
2. **RAGAS 未在当前环境安装**：`ragas` 不在根虚拟环境中，因此本文只根据代码和 `requirements-ragas.txt` 判断其用法；未实际运行 RAGAS 指标。启用 RAGAS 时还需要处理 `requirements-ragas.txt:24-25` 提到的 `instructor` 与 `openai>=3.23.0` 的解析冲突。
3. **Embedding 的哈希兜底**：`_HashEmbeddingFunction` 只在没有配置 `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` 时启用，属于本地开发兜底，不是语义检索方案；生产环境必须配置真实 embedding 服务。检索链路在 embedding 调用失败时跳过向量路，不回退到哈希向量（`backend/rag/retrieval.py:294-304`）。
4. **Milvus 部署形态**：代码同时支持本地 Milvus Lite 文件与远程 Milvus（`backend/rag/milvus_client.py:43-67`、`:392-421`）。具体部署使用哪一种取决于 `MILVUS_LITE_URI` / `MILVUS_URI` 配置，本文不根据某个本地 `.env` 值下结论。
5. **“未发现其他框架”的边界**：本文的结论基于对 `backend/` 的静态导入扫描和依赖声明；动态导入、运行时插件加载或第三方库内部的间接使用不在静态扫描范围内。不过，主链路的检索、融合、重排、切片和生成编排都能在仓库代码中逐行对应，未发现被外部 RAG 框架接管的入口。
6. **模型供应商**：`ChatOpenAI` 只是 OpenAI-compatible 客户端；实际供应商由 `DEEPSEEK_BASE_URL`、`TEXT_FALLBACK_BASE_URL`、`EMBEDDING_BASE_URL`、`RERANK_BASE_URL` 等配置决定，不能仅凭 `langchain-openai` 判断为“使用 OpenAI 官方服务”。

## 6. 关键证据索引

| 主题 | 文件 | 关键行 |
| --- | --- | --- |
| LangChain 唯一导入 | `backend/rag/llm.py` | 7 |
| ChatOpenAI 构造 | `backend/rag/llm.py` | 175-189、192-206、287-339 |
| 非流式/JSON 调用 | `backend/rag/llm.py` | 342-357、373-412 |
| 流式生成 | `backend/rag/llm.py` | 415-553、582-585 |
| RAG 路由 | `backend/rag/retrieval.py` | 162-181 |
| 查询规划（HyDE/改写/关键词） | `backend/rag/retrieval.py` | 184-212 |
| 多路 query 构造 | `backend/rag/retrieval.py` | 383-395 |
| 检索主流程 | `backend/rag/retrieval.py` | 215-353 |
| 并发召回 | `backend/rag/retrieval.py` | 251-304 |
| RRF | `backend/rag/retrieval.py` | 1353-1381 |
| 父子切片展开 | `backend/rag/retrieval.py` | 1534-1599 |
| BM25 实现 | `backend/rag/bm25.py` | 1-215 |
| BM25 召回接入 | `backend/rag/retrieval.py` | 441、579、948、1018 |
| Milvus 导入 | `backend/rag/milvus_client.py` | 64-65 |
| Milvus schema/index | `backend/rag/milvus_client.py` | 424-440 |
| 向量查询 | `backend/rag/milvus_client.py` | 724-765 |
| 向量写入 | `backend/rag/milvus_client.py` | 674-702 |
| Embedding HTTP | `backend/rag/milvus_client.py` | 332-385、583-671 |
| Rerank 主/兜底 | `backend/rag/rerank.py` | 31-112、151-203 |
| 最终 chunk 选择 | `backend/rag/rerank.py` | 206-219 |
| 聊天编排 | `backend/service/chat_service.py` | 783、847、990-994、1016、1074、1112-1118、1192、1322 |
| 上传入库 | `backend/service/knowledge_service.py` | 551-680 |
| 文本抽取/切片 | `backend/crud/knowledge_file.py` | 162、817、1683 |
| RAGAS 懒加载 | `backend/rag/ragas_eval.py` | 182-186 |
| RAGAS 指标 | `backend/rag/ragas_eval.py` | 204-208 |
| 依赖声明 | `backend/requirements.txt` | 11、14-19 |
| 可选 RAGAS 依赖 | `backend/requirements-ragas.txt` | 16、26 |
| 配置 | `backend/config.py` | 139-145、155、158-204、252-295、305-366、375-402、405-425、428-429、464-479、482、491 |
