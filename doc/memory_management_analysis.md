# 后端会话记忆管理分析

> 分析对象：`backend/` 下的会话记忆实现、配置、数据模型与测试。
> 行号对应分析时的文件内容。
> 本文中的「记忆」指**会话上下文记忆**（短期窗口 + 长期摘要），不包含 BM25 语料缓存、进程内单例缓存或学习 Trace 等运行时缓存。

## 1. 结论

本项目没有使用 LangChain Memory、向量记忆库或外部记忆服务。会话记忆是**自研的两层结构**，持久化在 MySQL 的 `conversations` 行上：

1. **短期记忆（最近对话窗口）**：读取当前问题之前、长期摘要指针之后的最近 `MEMORY_WINDOW_TURNS` 轮消息；超过 `MEMORY_RECENT_MAX_CHARS` 时，用 `memory` 阶段的 LLM 压缩成一条「近期记忆压缩」，失败时使用确定性文本兜底。
2. **长期记忆（会话摘要）**：`conversations.memory_summary` 保存累计摘要，`memory_summary_upto_message_id` 标记摘要已经覆盖到的消息 ID；每次 assistant 消息保存成功后，异步把滑出最近窗口的轮次合并进长期摘要，摘要超过 `MEMORY_SUMMARY_MAX_CHARS` 时再做一次压缩。

记忆在请求中的使用分三档：

| 使用位置 | 传入的记忆量 | 目的 |
| --- | --- | --- |
| 检索问题（`retrieval_question`） | 最多 1500 字符 | 只用于消解当前问题中的指代和省略，不作为事实依据 |
| RAG 路由模型（`decide_need_rag`） | 最多 1800 字符 | 辅助判断是否需要检索 |
| 最终回答生成 Prompt | 完整 `memory_context`（长期摘要最多 15000 字符 + 近期窗口最多 8000 字符） | 辅助理解指代、延续任务和用户偏好；事实优先级仍低于知识库上下文和当前问题 |

记忆更新是**异步旁路、尽力而为**：失败只记录日志和 Trace，不影响主回答；assistant 消息未保存（生成失败、空回答、提前断开）时不会触发摘要更新。所有同步 DB 操作都在工作线程中执行，Session 不跨线程、不跨 `await` 持有。

## 2. 数据模型与持久化

### 2.1 存储位置

记忆不单独建表，直接存放在会话行上：

```python
# backend/model/chat_models.py:77-79
memory_summary = Column(Text, default="")
memory_summary_upto_message_id = Column(Integer, default=0)
memory_updated_at = Column(DateTime, nullable=True)
```

字段含义：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `memory_summary` | `TEXT` | 长期记忆摘要文本；与近期窗口互补 |
| `memory_summary_upto_message_id` | `INTEGER`，默认 0 | 长期摘要已经覆盖到的最大消息 ID；短窗口查询用它排除已摘要消息 |
| `memory_updated_at` | `DATETIME` | 摘要最近一次写入时间；当前主要用于诊断，没有业务读取路径 |

旧库由启动迁移补列：

- `backend/database/session.py:82-90`：`ALTER TABLE conversations ADD COLUMN ...`
- `backend/database/session.py:93`：把 `memory_summary` 规范化为 `TEXT`。

### 2.2 会话状态快照

`stream_chat` 不在协程里直接持有 ORM `Conversation` 对象，而是在工作线程中把后续需要的字段复制成 `_ConversationState`：

```python
# backend/service/chat_service.py:612-620
class _ConversationState(NamedTuple):
    id: str
    title: str
    knowledge_base: _KnowledgeBaseRef
    memory_summary: str
    memory_summary_upto_message_id: int
    created: bool
```

快照构造在 `_load_or_create_conversation()` 中：

- `backend/service/chat_service.py:688-689`：`memory_summary=conversation.memory_summary or ""`、`memory_summary_upto_message_id=... or 0`。

这意味着：**一次请求使用请求开始时的记忆快照**；本轮请求期间异步更新产生的摘要，不会回写进当前回答的 Prompt。

### 2.3 消息本身是记忆的来源

短期窗口和摘要都从 `messages` 表读取；消息按 `id` 排序，而不是按 `created_at`：

- 短期窗口：`Message.id < current_message_id` 且 `Message.id > summary_upto`，按 `id` 倒序取最近一页，再反转成时间顺序（`backend/rag/memory_service.py:45-55`、`:77`）。
- 长期更新：读取 `Message.id > summary_upto`，按 `id` 升序（`backend/rag/memory_service.py:183-191`）。

`id` 是自增整数，能稳定表达先后关系；`created_at` 只用于展示，不参与记忆窗口排序。

## 3. 短期记忆：最近对话窗口

### 3.1 读取范围

```python
# backend/rag/memory_service.py:33-57
def _read_recent_messages(conversation_id, current_message_id, summary_upto):
    db = SessionLocal()
    try:
        return (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.id < current_message_id,
                Message.id > summary_upto,
            )
            .order_by(Message.id.desc())
            .limit(max(MEMORY_WINDOW_TURNS, 1) * 2)
            .all()
        )
    finally:
        db.close()
```

关键点：

- `current_message_id` 是**本轮用户消息**的 ID。查询使用 `< current_message_id`，所以当前问题不会重复进入「最近对话窗口」；当前问题由 `effective_question` 单独传给后续链路。
- `summary_upto` 来自会话快照的 `memory_summary_upto_message_id`；已经进入长期摘要的消息不会再次出现在短期窗口里。
- `limit(max(MEMORY_WINDOW_TURNS, 1) * 2)` 是**消息条数**上限，不是严格的「轮数」上限。默认 `MEMORY_WINDOW_TURNS=4`，最多取 8 条消息；后续按 user/assistant 分组时，实际轮数可能少于 `MEMORY_WINDOW_TURNS`。
- 整段同步 DB 工作（打开 Session、查询、关闭）都在 `_read_recent_messages` 内完成；调用方 `_build_recent_memory_text()` 用 `asyncio.to_thread` 进入（`backend/rag/memory_service.py:71-76`），因此 Session 不跨线程、不跨 `await`。

### 3.2 格式化为轮次

`_format_recent_memory_messages()` 把消息交给 `_group_message_texts_into_turns()`，把每条消息渲染为：

```text
用户：...
助手：...
```

并按轮次用空行连接：

- 分组：`_group_messages_into_turns()`，`backend/rag/memory_service.py:418-438`。
  - 遇到 `role == "user"` 开始新的一轮。
  - 后续 assistant 消息追加到当前轮。
  - 如果开头就是 assistant（没有 user），它会单独成为一轮。
- 渲染：`_group_message_texts_into_turns()`，`backend/rag/memory_service.py:441-453`。
  - 每条消息内容先 `_clip_text(content, 1200)`，即**单条消息最多 1200 字符**（硬编码）。
  - 空内容会被跳过。
  - 角色映射为「用户」/「助手」。
- 拼接：`_format_recent_memory_messages()`，`backend/rag/memory_service.py:107-115`；轮次之间用 `\n\n` 连接。

`_clip_text` 的行为见 `backend/service/utils_service.py:117-121`：超过上限时截到上限并在末尾加 `...`。

### 3.3 短期记忆超限时的压缩

如果拼接后的 `joined_text` 长度不超过 `MEMORY_RECENT_MAX_CHARS`（默认 8000），直接返回原文本：

```python
# backend/rag/memory_service.py:112-115
rendered_turns = ["\n".join(turn) for turn in turns]
joined_text = "\n\n".join(rendered_turns)
if len(joined_text) <= MEMORY_RECENT_MAX_CHARS:
    return joined_text
```

超过上限时：

1. 记录 Trace：`recent_memory_compaction_triggered`（`backend/rag/memory_service.py:117-129`）。
2. 调用 `_summarize_recent_memory(joined_text)`（`backend/rag/memory_service.py:130`）。
3. 成功则得到 `summary`，包装成 `近期记忆压缩：\n{summary}`，再按 `MEMORY_RECENT_MAX_CHARS` 截断后返回（`:131-142`）。
4. 失败/为空/被截断则调用 `_fallback_compact_recent_memory(rendered_turns)`（`backend/rag/memory_service.py:144-154`）。
5. 记录 Trace：`recent_memory_compacted` 或 `recent_memory_compaction_fallback`。

#### `_summarize_recent_memory()`

位置：`backend/rag/memory_service.py:475-516`。

- 如果 `MEMORY_API_KEY` 为空，直接警告并返回 `""`（`:476-478`），随后使用确定性兜底。
- System Prompt 要求保留：用户身份、偏好、约束、当前任务、明确问题、关键结论、待办；禁止逐字复述和编造（`:480-484`）。
- User Prompt 明确写「压缩长度上限：`MEMORY_RECENT_MAX_TOKENS` token」（`:485-489`）；默认 900 token。
- 通过 `call_chat_text_with_metadata(..., max_tokens=MEMORY_RECENT_MAX_TOKENS, stage="memory")` 调用模型（`:491-493`）。
- 如果 `finish_reason == "length"`，说明输出被 token 上限截断：记录 `recent_memory_compaction_truncated`，**拒绝该结果**并返回 `""`（`:494-512`）。这是 issue #264 的约束：半截摘要不能进入记忆。
- 正常完成时按 `MEMORY_RECENT_MAX_CHARS` 截断后返回（`:513`）。

#### `_fallback_compact_recent_memory()`

位置：`backend/rag/memory_service.py:464-472`。

- 固定前缀 `近期记忆压缩：\n`。
- 可用预算 = `MEMORY_RECENT_MAX_CHARS - len(prefix)`。
- 每轮预算 = `max(120, budget // 轮数)`，即每轮至少 120 字符（硬编码）。
- 每轮把换行替换成 ` / `，写成 `第N轮对话：...`，最后整体按 `MEMORY_RECENT_MAX_CHARS` 截断。

因此短期记忆在模型不可用时仍有一条**确定性、可重复**的兜底路径，不会因为摘要失败而让主回答中断。

### 3.4 短期记忆的配置

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `MEMORY_WINDOW_TURNS` | 4 | 最近保留的轮数；读取上限为 `max(值,1)*2` 条消息 |
| `MEMORY_RECENT_MAX_CHARS` | 8000 | 短期窗口注入/压缩前的字符上限（不单独落库） |
| `MEMORY_RECENT_MAX_TOKENS` | 900 | 近期记忆压缩模型的输出 token 上限 |

定义位置：`backend/config.py:482-488`；模板位置：`.env.example:197-201`。

## 4. 长期记忆：会话摘要

### 4.1 指针语义

`memory_summary_upto_message_id` 是长期摘要的「水位线」：

- 小于等于该 ID 的消息已经进入摘要，不再出现在短期窗口。
- 大于该 ID 的消息尚未被摘要，仍属于短期窗口的候选。
- 每次合并滑出窗口的轮次后，指针推进到这些轮次中最大的消息 ID（`backend/rag/memory_service.py:313`）。
- 二次压缩（摘要再压缩）不改变指针，因为压缩不新增覆盖范围（`_write_memory_summary_text(..., summary_upto_message_id=None)`，`backend/rag/memory_service.py:396`）。

### 4.2 触发时机

`stream_chat` 在 assistant 消息成功保存后，异步调度摘要更新：

```python
# backend/service/chat_service.py:1359-1361
try:
    _schedule_memory_summary_update(cid, trace.trace_id)
    await _safe_trace_add(
        trace,
        "memory_summary_update_scheduled",
        "_schedule_memory_summary_update",
        ...
    )
```

触发条件与语义：

- 仅在 `assistant_message_id is not None` 时执行（`backend/service/chat_service.py:1310-1360` 的 `if assistant_message_id is not None:` 分支）。
- 无论本轮是 RAG 还是直答都会执行；RAGAS 只在 `need_rag` 时调度，但记忆摘要更新不依赖 RAG。
- 生成失败、空回答、assistant 消息未落库时不会调度；测试 `tests/test_empty_model_response_259.py:127` 明确断言 `memory_summary_update_scheduled` 不出现。
- `_schedule_memory_summary_update()` 使用 `asyncio.create_task()`，不阻塞当前 SSE 流（`backend/rag/memory_service.py:157-168`）。

### 4.3 合并滑出窗口的轮次

入口：`_update_memory_summary_from_sliding_window()`，`backend/rag/memory_service.py:229-346`。

流程：

1. **读取会话摘要和未摘要消息**：`_read_conversation_window()`，`backend/rag/memory_service.py:171-194`。
   - 读 `Conversation.memory_summary` 和 `memory_summary_upto_message_id`。
   - 读 `Message.id > summary_upto` 的全部消息，按 `id` 升序。
   - 会话不存在返回 `None`，调用方记录 `memory_summary_update_skipped`（reason=`conversation_not_found`）。
2. **按轮分组**：`_group_messages_into_turns()`，`backend/rag/memory_service.py:251`。
3. **判断是否有轮次滑出最近窗口**：
   - `turns` 数量 `<= MEMORY_WINDOW_TURNS`：没有轮次滑出，跳过合并；只调用 `_compact_summary_if_needed()` 检查长期摘要是否过长（`backend/rag/memory_service.py:252-266`）。
   - 否则 `slipped_turns = turns[:-MEMORY_WINDOW_TURNS]`（`:268`），即保留最后 `MEMORY_WINDOW_TURNS` 轮在短期窗口，前面的轮次进入长期摘要。
4. **生成待合并 transcript**：`_format_messages_for_summary()`，`backend/rag/memory_service.py:270`、`:407-415`。
   - 每条消息内容截到 1200 字符（硬编码）。
   - 角色写为「用户」/「助手」。
   - transcript 为空则跳过（`:271-280`）。
5. **记录 Trace**：`memory_summary_update_triggered`，包含滑出轮数、消息数、旧摘要长度和窗口轮数（`:283-296`）。
6. **合并摘要**：`_summarize_conversation_memory(previous_summary, transcript)`（`:298-300`）。
   - 失败/为空时使用 `_fallback_merge_summary(previous_summary, transcript)`（`:301-310`）。
7. **推进指针并写回**：`summary_upto = max(message.id for message in slipped_messages)`，然后 `_write_memory_summary_text(conversation_id, summary_text, summary_upto)`（`:312-314`）。
8. **二次压缩检查**：`_compact_summary_if_needed()`（`:331`）。
9. **异常兜底**：整个流程包在 `try/except Exception` 中；失败只记录 `memory_summary_update_failed` 固定文案，不影响主回答（`:332-346`）。

### 4.4 长期摘要的生成 Prompt

`_summarize_conversation_memory()`：`backend/rag/memory_service.py:519-564`。

- `MEMORY_API_KEY` 为空时警告并返回 `""`（`:522-524`）。
- System Prompt：把给定记忆压缩成可用于后续追问理解的中文摘要；只保留用户身份、长期偏好、明确约束、当前任务背景、未解决问题和待办；不要保存知识库大段原文、不要编造、不要输出 Markdown 标题（`:526-531`）。
- User Prompt：明确「摘要长度上限：`MEMORY_SUMMARY_MAX_TOKENS` token」；输入包括「已有摘要」和「待合并记忆」（`:532-537`）；默认 900 token。
- 调用 `call_chat_text_with_metadata(..., max_tokens=MEMORY_SUMMARY_MAX_TOKENS, stage="memory")`（`:539-541`）。
- `finish_reason == "length"` 时拒绝截断结果，记录 `memory_summary_truncated` 并返回 `""`（`:542-560`）。
- 正常完成时按 `MEMORY_SUMMARY_MAX_CHARS` 截断（`:561`）。

### 4.5 长期摘要的二次压缩

`_compact_summary_if_needed()`：`backend/rag/memory_service.py:354-404`。

- 读取当前摘要（`_read_memory_summary()`，`:197-206`）。
- 摘要为空或长度 `<= MEMORY_SUMMARY_MAX_CHARS`（默认 15000）时直接返回（`:365-367`）。
- 超过上限时记录 `memory_summary_compaction_triggered`（`:369-380`），再调用
  `_summarize_conversation_memory("", summary)` 对摘要本身做压缩（`:381`）。
- 如果压缩失败/为空：使用 `_clip_text(summary, MEMORY_SUMMARY_MAX_CHARS)` 截断原摘要并写回，记录
  `memory_summary_compaction_failed`（`:382-393`）。
- 成功：写回压缩后的摘要（不改变指针），记录 `memory_summary_compacted`（`:395-404`）。

补充：

- `_maybe_compact_memory_summary()`（`:349-351`）只是对 `_compact_summary_if_needed()` 的包装，主要出现在测试中；生产路径通过 `_update_memory_summary_from_sliding_window()` 内的两次 `_compact_summary_if_needed()` 调用（`:265`、`:331`）触发。
- 长期摘要的字符上限和模型输出 token 上限是两套独立约束：Prompt 里写 token 上限，落库前按字符上限裁剪。

### 4.6 摘要写回与删除

`_write_memory_summary_text()`：`backend/rag/memory_service.py:209-226`。

- 打开独立 Session，按 `conversation_id` 查 `Conversation`。
- 会话不存在则静默返回（删除后的旁路任务不应报错）。
- 写入 `memory_summary`；如果传入 `summary_upto_message_id`，同时推进指针；总是写 `memory_updated_at = datetime.now()`。
- `commit()` 后关闭 Session。

消息删除会使摘要失效：

- `backend/crud/chat.py:828-835`：如果被删除的消息 ID 小于等于 `memory_summary_upto_message_id`，说明摘要可能包含已删内容，直接清空 `memory_summary`、把指针重置为 0、把 `memory_updated_at` 置空；下一轮重新生成。
- 如果删除的消息都在指针之后，摘要保持原样。
- 对应测试：`tests/test_message_delete_256.py:256-293`。

## 5. 记忆如何进入 RAG 与生成

### 5.1 构造 `memory_context`

`_build_memory_context()`：`backend/rag/memory_service.py:80-93`。

```python
summary = (conversation.memory_summary or "").strip()
sections = []
if summary:
    sections.append(f"长期摘要记忆：\n{_clip_text(summary, MEMORY_SUMMARY_MAX_CHARS)}")
if recent_text:
    sections.append(f"最近对话窗口：\n{recent_text}")
return "\n\n".join(sections).strip()
```

- 长期摘要部分最多 `MEMORY_SUMMARY_MAX_CHARS`（默认 15000）字符；但写入时已经按同一上限裁剪过。
- 近期窗口部分由 `_build_recent_memory_text()` 产生，最多 `MEMORY_RECENT_MAX_CHARS`（默认 8000）字符。
- 两段都为空时返回 `""`。

### 5.2 构造检索问题

`_build_memory_aware_retrieval_question()`：`backend/rag/memory_service.py:96-104`。

```python
if not memory_context:
    return question
compact_memory = _clip_text(memory_context, 1500)
return (
    "以下会话记忆仅用于消解当前问题中的指代和省略，不作为事实依据。\n"
    f"{compact_memory}\n\n"
    f"当前检索问题：{question}"
)
```

- 检索问题最多携带 1500 字符记忆，避免把整段摘要塞进查询。
- 明确告诉模型/检索链：记忆只用于指代消解，不作为事实依据。

### 5.3 `stream_chat` 的接入顺序

`backend/service/chat_service.py:982-1014`：

1. `recent_text = await _build_recent_memory_text(cid, current_message_id=user_message_id, summary_upto=conversation.memory_summary_upto_message_id, trace_id=...)`（`:982-989`）。
2. `memory_context = _build_memory_context(conversation, recent_text=recent_text)`（`:990-993`）。
3. `retrieval_question = _build_memory_aware_retrieval_question(effective_question, memory_context)`（`:994`）。
4. 记录 Trace `memory_built`，包含 `memory_summary`、`recent_text`、`memory_context`、`retrieval_question`，以及 `memory_used`、`used_for_retrieval`、`window_turns`（`:995-1014`）。

随后：

- RAG 路由：`decide_need_rag(effective_question, memory_context, knowledge_base.name, body.attachments)`（`backend/service/chat_service.py:1016-1021`）。`decide_need_rag()` 内部把记忆截到 1800 字符再放进 payload（`backend/rag/retrieval.py:169-174`）。
- 检索：`retrieve_knowledge(retrieval_question, ...)`（`backend/service/chat_service.py:1074-1082`），使用带记忆的检索问题。
- 生成：`stream_rag_answer(effective_question, context, memory_context, trace, use_rag=need_rag)`（`backend/service/chat_service.py:1192-1198`），生成 Prompt 使用完整 `memory_context`。

### 5.4 生成 Prompt 中的记忆位置

`build_answer_messages()`：`backend/rag/llm.py:554-579`。

RAG 模式：

```text
信息优先级为：企业知识库上下文 > 当前用户问题 > 会话记忆。
会话记忆只能用于理解指代、延续任务和用户偏好，不能替代知识库事实依据。
```

User Prompt 分区：

```text
用户问题：{question}

会话记忆：
{memory_context 或 （无）}

企业知识库上下文：
{context 或 （未检索到相关知识库内容）}
```

直答模式（`use_rag=False`）：

```text
用户问题：{question}

会话记忆：
{memory_context 或 （无）}

回答要求：直接回答当前问题，不要引用知识库上下文，也不要提到检索过程。
```

因此：**短期窗口和长期摘要最终作为「会话记忆」进入生成 Prompt**；知识库上下文和当前问题优先于记忆，记忆不能替代知识库事实。

## 6. 配置项

### 6.1 记忆配置

| 配置 | 默认 | 作用 | 定义位置 | 模板位置 |
| --- | --- | --- | --- | --- |
| `MEMORY_WINDOW_TURNS` | 4 | 最近窗口保留轮数；短期读取上限为 `max(值,1)*2` 条消息 | `config.py:482` | `.env.example:197` |
| `MEMORY_SUMMARY_MAX_TOKENS` | 900 | 长期摘要模型的输出 token 上限 | `config.py:485` | `.env.example:198` |
| `MEMORY_RECENT_MAX_TOKENS` | 900 | 近期记忆压缩模型的输出 token 上限 | `config.py:486` | `.env.example:199` |
| `MEMORY_SUMMARY_MAX_CHARS` | 15000 | 长期摘要落库/注入的字符上限 | `config.py:487` | `.env.example:200` |
| `MEMORY_RECENT_MAX_CHARS` | 8000 | 近期窗口注入/压缩的字符上限（不单独落库） | `config.py:488` | `.env.example:201` |

### 6.2 LLM 阶段配置

记忆摘要通过 `stage="memory"` 调用模型：

- `MEMORY_BASE_URL`、`MEMORY_API_KEY`、`MEMORY_MODEL`：`backend/config.py:173-175`。
- 继承关系：`memory` 阶段默认继承 `DEEPSEEK_*`；如果显式配置了 `MEMORY_*`，则按阶段覆盖使用（`backend/rag/llm.py:212-229`、`:287-339`）。
- `MEMORY_API_KEY` 为空时，`_summarize_recent_memory()` / `_summarize_conversation_memory()` 直接返回空字符串，系统改用确定性兜底（`backend/rag/memory_service.py:476-478`、`:522-524`）。

### 6.3 硬编码预算

除了配置项，代码里还有几个固定值：

- **单条消息最大 1200 字符**：`_format_messages_for_summary()`（`backend/rag/memory_service.py:410`）和 `_group_message_texts_into_turns()`（`:446`）。
- **短期兜底每轮最少 120 字符**：`_fallback_compact_recent_memory()`（`backend/rag/memory_service.py:467`）。
- **检索问题最多 1500 字符记忆**：`_build_memory_aware_retrieval_question()`（`backend/rag/memory_service.py:99`）。
- **路由模型最多 1800 字符记忆**：`decide_need_rag()`（`backend/rag/retrieval.py:171` 的 `_clip(memory_context, 1800)`）。

## 7. 时序与失效流程

### 7.1 一次正常问答

```text
用户消息保存（user_message_id）
  -> _build_recent_memory_text(cid, current_message_id=user_message_id, summary_upto=当前指针)
       -> 只取 id < user_message_id 且 id > summary_upto 的最近消息
       -> 按轮分组；超过 MEMORY_RECENT_MAX_CHARS 时压缩/兜底
  -> _build_memory_context(长期摘要 + 最近窗口)
  -> _build_memory_aware_retrieval_question（最多 1500 字符记忆）
  -> decide_need_rag（最多 1800 字符记忆）
  -> retrieve_knowledge(retrieval_question)
  -> stream_rag_answer(effective_question, context, memory_context)
  -> assistant 消息保存成功
  -> _schedule_memory_summary_update(cid, trace_id)  [异步任务]
       -> 读 id > summary_upto 的消息
       -> 分组；如果轮数 > MEMORY_WINDOW_TURNS
            -> 把滑出窗口的轮次合并进长期摘要
            -> 指针推进到滑出轮次的最大消息 ID
            -> 检查长期摘要是否超过 MEMORY_SUMMARY_MAX_CHARS，必要时二次压缩
       -> 如果轮数 <= MEMORY_WINDOW_TURNS
            -> 只检查长期摘要是否需要二次压缩
```

### 7.2 失败与不触发的情况

- **生成失败/空回答**：没有 assistant 消息落库，不调度摘要更新（`tests/test_empty_model_response_259.py:127`）。
- **摘要模型失败/Key 未配置**：`_summarize_*` 返回空字符串；短期使用 `_fallback_compact_recent_memory()`，长期使用 `_fallback_merge_summary()` 或截断原摘要。
- **摘要被 token 上限截断**：`finish_reason == "length"` 时拒绝结果，短期走确定性兜底，长期走 fallback merge；不会把半截摘要写入记忆（`tests/test_memory_summary_truncation_264.py`）。
- **会话被删除**：`_read_conversation_window()` 返回 `None`，更新跳过；`_write_memory_summary_text()` 找不到会话时静默返回。
- **消息删除覆盖摘要**：清空摘要、指针和 `memory_updated_at`，下一轮重新生成（`backend/crud/chat.py:828-835`）。
- **调度失败（没有事件循环）**：`_schedule_memory_summary_update()` 捕获 `RuntimeError`，记录 `memory_summary_update_schedule_failed`，主回答不受影响（`backend/rag/memory_service.py:157-168`）。
- **更新任务内部异常**：记录 `memory_summary_update_failed` 固定文案，异常原文只进日志（`backend/rag/memory_service.py:332-346`；`tests/test_internal_error_no_echo.py:793-819`）。

## 8. 并发、线程与可观测性

### 8.1 线程边界

记忆模块的每条同步 DB 路径都遵循同一规则：

- 在 `_read_recent_messages()` 内打开/查询/关闭 Session（`backend/rag/memory_service.py:33-57`）。
- 在 `_read_conversation_window()` 内打开/查询/关闭 Session（`:171-194`）。
- 在 `_read_memory_summary()` 内打开/查询/关闭 Session（`:197-206`）。
- 在 `_write_memory_summary_text()` 内打开/写入/提交/关闭 Session（`:209-226`）。
- 调用方通过 `asyncio.to_thread(...)` 进入这些函数（如 `:71-76`、`:238`、`:314`、`:362`）。

回归测试 `tests/test_sync_session_off_loop_187.py:206-256` 覆盖：

- `_maybe_compact_memory_summary()`：读摘要和写回不在事件循环线程上。
- `_update_memory_summary_from_sliding_window()`：滑出窗口更新不在事件循环线程上。
- `_build_recent_memory_text()`：SSE 主链路上的最近窗口查询不在事件循环线程上。

测试不仅检查心跳，还检查每个 Session 的「打开、使用、关闭」发生在同一个工作线程上。

### 8.2 异步调度

- `_schedule_memory_summary_update()` 使用 `asyncio.create_task()` 启动后台任务，不 `await`（`backend/rag/memory_service.py:157-159`）。
- 当前代码没有保存 Task 的强引用；按 `asyncio` 文档，事件循环对 Task 只保留弱引用，任务在极小概率下可能被垃圾回收。这是一个需要留意的运行时边界，不是现有测试覆盖的行为。
- 记忆更新可能在本轮 SSE 完成之后才结束，Trace 中会留下 `memory_summary_update_scheduled` 以及后续的 `memory_summary_update_*` 事件。

### 8.3 Trace 事件

请求内的记忆构造：

- `memory_built`：`backend/service/chat_service.py:995-1014`，包含 `memory_summary`、`recent_text`、`memory_context`、`retrieval_question` 和 `window_turns` 等信息。

摘要更新：

- `memory_summary_update_scheduled`：`backend/service/chat_service.py:1360-1380`。
- `memory_summary_update_triggered`、`memory_summary_updated`、`memory_summary_update_fallback`、`memory_summary_update_failed`、`memory_summary_update_skipped`：`backend/rag/memory_service.py:240-346`。
- `memory_summary_compaction_triggered`、`memory_summary_compacted`、`memory_summary_compaction_failed`：`backend/rag/memory_service.py:369-404`。
- `recent_memory_compaction_triggered`、`recent_memory_compacted`、`recent_memory_compaction_fallback`：`backend/rag/memory_service.py:117-154`。
- `recent_memory_compaction_truncated`、`memory_summary_truncated`：`backend/rag/memory_service.py:494-512`、`:542-560`。

注意：`backend/rag/learning_trace.py:22-30` 把 `recent_text`、`memory_context`、`memory_summary`、`conv.memory_summary`、`retrieval_question`、`transcript` 列为 `FULL_TEXT_TRACE_KEYS`，`sanitize_trace_value()` 对这些键**不做截断**（`:45-48`）。也就是说，记忆正文可能完整写入 `chat_trace_sessions.events`，用于调试和回放；这属于需要关注的存储/隐私边界。

## 9. 测试证据

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_memory_context.py` | `_build_memory_context()` 拼接长期摘要与最近窗口；空段落跳过；`_group_message_texts_into_turns()` 保留 falsy 内容 |
| `tests/test_memory_summary_truncation_264.py` | 摘要必须按 token 预算生成；`finish_reason=length` 的半截摘要不得返回/落库；短期压缩失败走确定性兜底；`MEMORY_*_MAX_TOKENS` 与 `.env.example` 对齐 |
| `tests/test_sync_session_off_loop_187.py:206-256` | 最近窗口读取、滑出窗口摘要更新、长期摘要压缩的同步 DB 段都在工作线程；Session 不跨线程 |
| `tests/test_message_delete_256.py:256-293` | 删除消息覆盖摘要时清空摘要/指针/更新时间；删除区域在指针之后时摘要保留 |
| `tests/test_chat_service_retrieval.py:61`、`:247` | `memory_built` 阶段存在；assistant 保存后按真实参数调度 `_schedule_memory_summary_update(cid, trace_id)` |
| `tests/test_internal_error_no_echo.py:250-258`、`:793-819` | 记忆调度失败/更新失败时，Trace 只写固定文案，不回显异常原文；服务端日志仍保留异常 |
| `tests/test_empty_model_response_259.py:127` | 空回答/生成失败时不出现 `memory_summary_update_scheduled` |

## 10. 边界、风险与未确认项

1. **记忆是「会话级」的，不是「用户级」的**：长期摘要存放在 `conversations` 行上，只服务于该会话；没有跨会话的全局用户记忆，也没有向量化记忆库。
2. **记忆更新是异步旁路**：当前回答使用的是请求开始时的快照；本轮 assistant 落库后调度的摘要更新不会改变本轮已经生成的回答，只影响后续请求。
3. **并发摘要更新没有显式锁或版本号**：`_schedule_memory_summary_update()` 每次 `create_task` 一个更新任务；同一会话若同时有多个请求完成，可能出现多个任务同时读同一摘要、分别写回，最后写入者覆盖前一个结果，指针也可能回退。代码中没有看到会话级锁、版本号或 CAS 校验；`_write_memory_summary_text()` 是直接赋值 + `commit()`（`backend/rag/memory_service.py:209-226`）。这是一个潜在竞态，现有测试未覆盖。
4. **Task 未被强引用**：`asyncio.create_task()` 的返回值没有保存；按 Python 文档，事件循环只保留弱引用。该边界可能影响后台任务的可靠执行，现有测试没有覆盖。
5. **`MEMORY_WINDOW_TURNS` 没有下界校验**：`_read_recent_messages()` 用 `max(值,1)` 兜底，但 `_update_memory_summary_from_sliding_window()` 直接用原始值做 `turns[:-MEMORY_WINDOW_TURNS]`。如果配置为 0，`turns[:-0]` 为空，随后因 transcript 为空而跳过长期摘要更新；这是基于代码推导的行为，未见到专门测试。
6. **`_maybe_compact_memory_summary()` 不是生产主入口**：它是对 `_compact_summary_if_needed()` 的测试包装；生产路径通过 `_update_memory_summary_from_sliding_window()` 内的两次 `_compact_summary_if_needed()` 调用触发（`backend/rag/memory_service.py:265`、`:331`）。
7. **记忆正文会进入 Trace**：`FULL_TEXT_TRACE_KEYS` 中的记忆键不截断，可能完整写入 `chat_trace_sessions.events`；若部署把 Trace 长期保留，需要评估存储与隐私影响（`backend/rag/learning_trace.py:22-30`、`:45-48`）。
8. **删除与摘要更新的竞态**：消息删除会清空覆盖到的摘要（`backend/crud/chat.py:828-835`），但摘要更新任务没有与删除路径共享锁。若更新任务在删除前读取、删除后写回，理论上可能把已删内容重新写进摘要。现有测试覆盖了「删除后摘要被清空」，但没有覆盖「更新任务与删除并发」的时序。
9. **记忆更新依赖 assistant 消息落库**：生成失败、空回答、客户端提前断开导致 assistant 未保存时，不会触发摘要更新；这是由 `assistant_message_id is not None` 分支决定的（`backend/service/chat_service.py:1310-1361`）。
10. **`memory_updated_at` 当前没有业务读取路径**：它会被写入（`backend/rag/memory_service.py:223`），但未发现用于过期判断或排序的读取方；它更像诊断字段。

## 11. 关键证据索引

| 主题 | 文件 | 关键行 |
| --- | --- | --- |
| 记忆字段模型 | `backend/model/models.py` | 77-79 |
| 记忆字段迁移 | `backend/database/session.py` | 82-90、93 |
| 会话状态快照 | `backend/service/chat_service.py` | 612-620、688-689 |
| 最近窗口读取 | `backend/rag/memory_service.py` | 33-57 |
| 最近窗口构造 | `backend/rag/memory_service.py` | 60-77 |
| 记忆上下文拼接 | `backend/rag/memory_service.py` | 80-93 |
| 检索问题注入 | `backend/rag/memory_service.py` | 96-104 |
| 最近窗口格式化/压缩 | `backend/rag/memory_service.py` | 107-154 |
| 摘要更新调度 | `backend/rag/memory_service.py` | 157-168 |
| 摘要读取/写回 | `backend/rag/memory_service.py` | 171-226 |
| 滑出窗口合并 | `backend/rag/memory_service.py` | 229-346 |
| 长期摘要二次压缩 | `backend/rag/memory_service.py` | 349-404 |
| 消息格式化/轮次分组 | `backend/rag/memory_service.py` | 407-453 |
| 确定性兜底 | `backend/rag/memory_service.py` | 456-472 |
| 近期记忆摘要 Prompt | `backend/rag/memory_service.py` | 475-516 |
| 长期摘要 Prompt | `backend/rag/memory_service.py` | 519-564 |
| 记忆接入聊天主链路 | `backend/service/chat_service.py` | 982-1014、1016-1021、1074-1082、1192-1198 |
| 摘要更新调度点 | `backend/service/chat_service.py` | 1360-1380 |
| 生成 Prompt 中的记忆 | `backend/rag/llm.py` | 554-579 |
| 路由模型记忆裁剪 | `backend/rag/retrieval.py` | 169-174 |
| 删除消息清空摘要 | `backend/crud/chat.py` | 828-835 |
| 记忆配置 | `backend/config.py` | 173-175、482-488 |
| 配置模板 | `.env.example` | 197-201 |
| Trace 全文键 | `backend/rag/learning_trace.py` | 22-30、45-48 |
| 线程边界测试 | `tests/test_sync_session_off_loop_187.py` | 206-256 |
| 摘要截断测试 | `tests/test_memory_summary_truncation_264.py` | 1-210 |
| 消息删除测试 | `tests/test_message_delete_256.py` | 256-293 |
| 主链路集成测试 | `tests/test_chat_service_retrieval.py` | 61、247 |
| 错误文本不泄漏测试 | `tests/test_internal_error_no_echo.py` | 250-258、793-819 |
