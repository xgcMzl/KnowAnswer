<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Promotion } from '@element-plus/icons-vue'

import ConversationSidebar from '@/components/ConversationSidebar.vue'
import MessageBubble from '@/components/MessageBubble.vue'
import StreamingBubble from '@/components/StreamingBubble.vue'
import { useChatStore } from '@/stores/chat'
import { useKnowledgeStore } from '@/stores/knowledge'
import type { Conversation } from '@/types'

const chat = useChatStore()
const knowledge = useKnowledgeStore()

const selectedKbId = ref<number | null>(null)
const input = ref('')
const scrollRef = ref<HTMLElement | null>(null)

const suggestions = [
  '这份文档的主要内容是什么？',
  '总结一下关键结论',
  '文档里有哪些注意事项？',
]

const activeConversation = computed<Conversation | null>(
  () => chat.conversations.find((item) => item.id === chat.activeId) ?? null,
)

const selectedKbName = computed(() => {
  if (selectedKbId.value === null) return '全部知识库'
  return (
    knowledge.bases.find((item) => item.id === selectedKbId.value)?.name ?? '全部知识库'
  )
})

onMounted(async () => {
  try {
    await Promise.all([chat.loadConversations(), knowledge.loadBases()])
    if (chat.activeId === null && chat.conversations.length > 0) {
      await chat.activate(chat.conversations[0].id)
    }
  } catch (error) {
    ElMessage.error((error as Error).message || '加载数据失败')
  }
})

watch(
  () => chat.activeId,
  (id) => {
    if (id === null) {
      selectedKbId.value = null
      return
    }
    const conversation = chat.conversations.find((item) => item.id === id)
    selectedKbId.value = conversation?.knowledge_base_id ?? null
    scrollToBottom()
  },
)

watch(() => chat.messages.length, scrollToBottom)
watch(() => chat.streamContent, scrollToBottom)

function scrollToBottom(): void {
  void nextTick(() => {
    const element = scrollRef.value
    if (element) {
      element.scrollTop = element.scrollHeight
    }
  })
}

async function createConversation(): Promise<void> {
  if (chat.streaming) return
  try {
    await chat.newConversation(selectedKbId.value)
    await chat.loadConversations()
  } catch (error) {
    ElMessage.error((error as Error).message || '创建会话失败')
  }
}

async function sendMessage(): Promise<void> {
  const text = input.value.trim()
  if (!text || chat.streaming) return
  input.value = ''
  try {
    await chat.send(text, selectedKbId.value)
  } catch (error) {
    ElMessage.error((error as Error).message || '发送失败')
  }
  if (chat.streamError) {
    ElMessage.warning(chat.streamError)
  }
}

function onKeydown(event: Event | KeyboardEvent): void {
  if (!(event instanceof KeyboardEvent)) return
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    void sendMessage()
  }
}

function useSuggestion(text: string): void {
  input.value = text
  void sendMessage()
}
</script>

<template>
  <div class="chat-view">
    <ConversationSidebar @new="createConversation" />

    <section class="chat-panel">
      <header class="chat-header">
        <div class="header-title">
          <h2>{{ activeConversation?.title || (chat.activeId ? '对话' : '新对话') }}</h2>
          <span>{{ chat.activeId ? `${chat.messages.length} 条消息` : '从下方输入框开始提问' }}</span>
        </div>

        <el-select
          v-model="selectedKbId"
          clearable
          placeholder="全部知识库"
          class="kb-select"
        >
          <el-option
            v-for="base in knowledge.bases"
            :key="base.id"
            :label="base.name"
            :value="base.id"
          />
        </el-select>
      </header>

      <div ref="scrollRef" v-loading="chat.loadingMessages" class="messages">
        <div v-if="!chat.messages.length && !chat.streaming" class="empty-state">
          <div class="empty-mark">K</div>
          <h3>开始你的知识库问答</h3>
          <p>先在“知识库”页上传文档，再回到这里提问。</p>
          <div class="suggestion-list">
            <button
              v-for="suggestion in suggestions"
              :key="suggestion"
              type="button"
              @click="useSuggestion(suggestion)"
            >
              {{ suggestion }}
            </button>
          </div>
        </div>

        <MessageBubble
          v-for="message in chat.messages"
          :key="message.id"
          :message="message"
        />
        <StreamingBubble
          v-if="chat.streaming"
          :content="chat.streamContent"
          :sources="chat.streamSources"
        />
      </div>

      <footer class="composer">
        <el-input
          v-model="input"
          type="textarea"
          :autosize="{ minRows: 1, maxRows: 6 }"
          resize="none"
          placeholder="输入问题，Enter 发送，Shift + Enter 换行"
          :disabled="chat.streaming"
          @keydown="onKeydown"
        />
        <div class="composer-actions">
          <span class="composer-hint">检索范围：{{ selectedKbName }}</span>
          <el-button
            v-if="chat.streaming"
            type="danger"
            plain
            @click="chat.stop"
          >
            停止生成
          </el-button>
          <el-button
            v-else
            type="primary"
            :icon="Promotion"
            :disabled="!input.trim()"
            @click="sendMessage"
          >
            发送
          </el-button>
        </div>
      </footer>
    </section>
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  height: 100%;
  overflow: hidden;
}

.chat-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  background: var(--ka-panel);
}

.chat-header {
  height: 68px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 26px;
  border-bottom: 1px solid var(--ka-border);
  background: #fff;
}

.header-title h2 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #0f172a;
}

.header-title span {
  display: block;
  margin-top: 3px;
  font-size: 12px;
  color: var(--ka-muted);
}

.kb-select {
  width: 210px;
}

.messages {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 26px 30px 10px;
  background: linear-gradient(180deg, #fafbff, #f5f7fb 220px);
}

.empty-state {
  max-width: 560px;
  margin: 9vh auto 0;
  text-align: center;
}

.empty-mark {
  width: 58px;
  height: 58px;
  margin: 0 auto 16px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  font-size: 26px;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  box-shadow: 0 12px 26px rgba(99, 102, 241, 0.3);
}

.empty-state h3 {
  margin: 0 0 8px;
  font-size: 19px;
  color: #111827;
}

.empty-state p {
  margin: 0 0 20px;
  font-size: 13.5px;
  color: var(--ka-muted);
}

.suggestion-list {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  justify-content: center;
}

.suggestion-list button {
  padding: 8px 14px;
  border-radius: 20px;
  border: 1px solid #dcdff0;
  background: #fff;
  color: #4b5563;
  font-size: 12.5px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.suggestion-list button:hover {
  border-color: var(--ka-primary);
  color: var(--ka-primary);
  background: #f5f6ff;
}

.composer {
  flex-shrink: 0;
  padding: 14px 26px 18px;
  border-top: 1px solid var(--ka-border);
  background: #fff;
}

.composer :deep(.el-textarea__inner) {
  padding: 11px 13px;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.7;
  box-shadow: 0 0 0 1px var(--ka-border) inset;
}

.composer :deep(.el-textarea__inner:focus) {
  box-shadow: 0 0 0 1px var(--ka-primary) inset;
}

.composer-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 14px;
  margin-top: 9px;
}

.composer-hint {
  margin-right: auto;
  font-size: 12px;
  color: #9ca3af;
}

@media (max-width: 900px) {
  .messages {
    padding: 18px 16px 8px;
  }

  .chat-header,
  .composer {
    padding-left: 16px;
    padding-right: 16px;
  }

  .kb-select {
    width: 150px;
  }
}
</style>
