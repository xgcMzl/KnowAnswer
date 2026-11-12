import { ref } from 'vue'
import { defineStore } from 'pinia'

import * as api from '@/api/chat'
import { ApiError } from '@/api/http'
import type { Conversation, Message, Source } from '@/types'

export const useChatStore = defineStore('chat', () => {
  const conversations = ref<Conversation[]>([])
  const activeId = ref<number | null>(null)
  const messages = ref<Message[]>([])
  const loadingConversations = ref(false)
  const loadingMessages = ref(false)
  const streaming = ref(false)
  const streamContent = ref('')
  const streamSources = ref<Source[]>([])
  const streamError = ref<string | null>(null)

  let controller: AbortController | null = null

  async function loadConversations(): Promise<void> {
    loadingConversations.value = true
    try {
      const { items } = await api.listConversations()
      conversations.value = items
    } finally {
      loadingConversations.value = false
    }
  }

  async function activate(id: number): Promise<void> {
    activeId.value = id
    await loadMessages(id)
  }

  async function loadMessages(id: number): Promise<void> {
    loadingMessages.value = true
    try {
      const { items } = await api.listMessages(id)
      messages.value = items
    } finally {
      loadingMessages.value = false
    }
  }

  async function newConversation(
    knowledgeBaseId: number | null,
    title = '',
  ): Promise<Conversation> {
    const conversation = await api.createConversation(knowledgeBaseId, title)
    conversations.value.unshift(conversation)
    activeId.value = conversation.id
    messages.value = []
    return conversation
  }

  async function renameConversation(id: number, title: string): Promise<void> {
    const updated = await api.updateConversation(id, title)
    const index = conversations.value.findIndex((item) => item.id === id)
    if (index >= 0) {
      conversations.value[index] = updated
    }
  }

  async function removeConversation(id: number): Promise<void> {
    await api.deleteConversation(id)
    conversations.value = conversations.value.filter((item) => item.id !== id)
    if (activeId.value === id) {
      const next = conversations.value[0]
      if (next) {
        await activate(next.id)
      } else {
        activeId.value = null
        messages.value = []
      }
    }
  }

  async function send(question: string, knowledgeBaseId: number | null): Promise<void> {
    if (streaming.value) return
    const text = question.trim()
    if (!text) return

    let conversationId = activeId.value
    if (conversationId === null) {
      const conversation = await newConversation(knowledgeBaseId)
      conversationId = conversation.id
    }

    const tempUser: Message = {
      id: -Date.now(),
      conversation_id: conversationId,
      role: 'user',
      content: text,
      sources: null,
      created_at: new Date().toISOString(),
    }
    messages.value.push(tempUser)

    streaming.value = true
    streamContent.value = ''
    streamSources.value = []
    streamError.value = null
    controller = new AbortController()

    try {
      await api.streamChat(
        {
          conversation_id: conversationId,
          question: text,
          knowledge_base_id: knowledgeBaseId,
        },
        {
          onSources: (sources) => {
            streamSources.value = sources
          },
          onDelta: (delta) => {
            streamContent.value += delta
          },
          onDone: (event) => {
            messages.value.push({
              id: event.message_id,
              conversation_id: event.conversation_id,
              role: 'assistant',
              content: streamContent.value,
              sources: streamSources.value.length ? [...streamSources.value] : null,
              created_at: new Date().toISOString(),
            })
          },
        },
        controller.signal,
      )
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        streamError.value = '已停止生成，本轮回答未保存'
      } else if (error instanceof ApiError) {
        streamError.value = error.message
      } else {
        streamError.value = '生成失败，请稍后重试'
      }
    } finally {
      streaming.value = false
      controller = null
      streamContent.value = ''
      streamSources.value = []
      try {
        await loadMessages(conversationId)
        await loadConversations()
      } catch {
        // 同步失败不影响已展示的内容
      }
    }
  }

  function stop(): void {
    controller?.abort()
  }

  return {
    conversations,
    activeId,
    messages,
    loadingConversations,
    loadingMessages,
    streaming,
    streamContent,
    streamSources,
    streamError,
    loadConversations,
    activate,
    loadMessages,
    newConversation,
    renameConversation,
    removeConversation,
    send,
    stop,
  }
})
