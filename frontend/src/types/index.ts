export interface KnowledgeBase {
  id: number
  name: string
  description: string
  file_count: number
  created_at: string
  updated_at: string
}

export type FileStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface KnowledgeFile {
  id: number
  knowledge_base_id: number
  file_name: string
  file_ext: string
  file_size: number
  status: FileStatus
  chunk_count: number
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface Conversation {
  id: number
  knowledge_base_id: number | null
  title: string
  created_at: string
  updated_at: string
}

export interface Source {
  index: number
  node_id: string
  /** 父块 ID；旧消息或未启用父子切片时为空 */
  parent_id?: string | null
  parent_index?: number | null
  file_id: number | null
  file_name: string
  knowledge_base_id: number | null
  page_label: string | null
  chunk_index: number | null
  /** 混合检索下是 RRF 融合分，纯向量检索下是余弦相似度 */
  score: number | null
  score_type?: 'rrf' | 'dense' | null
  /** 命中的子块原文（text 为父块全文时用于展示命中位置） */
  matched_text?: string | null
  text: string
}

export interface Message {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string
  sources: Source[] | null
  created_at: string
}

export interface HealthResponse {
  status: string
  database: string
  milvus: { status: string; detail?: string }
  embedding_model: string
  llm: string
}

export interface ChatStreamPayload {
  conversation_id: number
  question: string
  knowledge_base_id?: number | null
}

export type StreamEvent =
  | { type: 'sources'; sources: Source[] }
  | { type: 'delta'; content: string }
  | { type: 'done'; conversation_id: number; message_id: number }
  | { type: 'error'; code: string; message: string }
