import { API_BASE, ApiError, jsonRequest, request } from './http'
import type {
  ChatStreamPayload,
  Conversation,
  Intent,
  Message,
  StreamEvent,
} from '@/types'

export function listConversations(): Promise<{ items: Conversation[] }> {
  return request('/api/conversations')
}

export function createConversation(
  knowledgeBaseId: number | null,
  title = '',
): Promise<Conversation> {
  return jsonRequest('POST', '/api/conversations', {
    knowledge_base_id: knowledgeBaseId,
    title,
  })
}

export function updateConversation(id: number, title: string): Promise<Conversation> {
  return jsonRequest('PATCH', `/api/conversations/${id}`, { title })
}

export function deleteConversation(id: number): Promise<void> {
  return jsonRequest('DELETE', `/api/conversations/${id}`)
}

export function listMessages(conversationId: number): Promise<{ items: Message[] }> {
  return request(`/api/conversations/${conversationId}/messages`)
}

export interface StreamHandlers {
  onIntent?: (intent: Intent) => void
  onSources?: (sources: import('@/types').Source[]) => void
  onDelta?: (content: string) => void
  onDone?: (event: Extract<StreamEvent, { type: 'done' }>) => void
}

/** 读取 POST SSE 流；收到 error 帧时抛出 ApiError。 */
export async function streamChat(
  payload: ChatStreamPayload,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    try {
      const data = await response.json()
      const detail = data?.error
      throw new ApiError(
        detail?.message || `请求失败（HTTP ${response.status}）`,
        detail?.code || 'HTTP_ERROR',
        response.status,
      )
    } catch (error) {
      if (error instanceof ApiError) throw error
      throw new ApiError(`请求失败（HTTP ${response.status}）`, 'HTTP_ERROR', response.status)
    }
  }

  if (!response.body) {
    throw new ApiError('当前浏览器不支持流式响应', 'STREAM_UNSUPPORTED')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const chunk = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')

      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data:')) continue
        const raw = line.slice(5).trim()
        if (!raw) continue
        if (raw === '[DONE]') return

        let event: StreamEvent
        try {
          event = JSON.parse(raw) as StreamEvent
        } catch {
          continue
        }

        if (event.type === 'intent') {
          handlers.onIntent?.(event.intent)
        } else if (event.type === 'sources') {
          handlers.onSources?.(event.sources)
        } else if (event.type === 'delta') {
          handlers.onDelta?.(event.content)
        } else if (event.type === 'done') {
          handlers.onDone?.(event)
        } else if (event.type === 'error') {
          throw new ApiError(event.message, event.code)
        }
      }
    }
  }
}
