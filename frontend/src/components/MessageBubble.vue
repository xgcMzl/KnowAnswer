<script setup lang="ts">
import { computed } from 'vue'
import { ElMessage } from 'element-plus'

import MarkdownContent from './MarkdownContent.vue'
import SourcePanel from './SourcePanel.vue'
import { formatTime } from '@/utils/format'
import type { Message } from '@/types'

const props = defineProps<{ message: Message }>()

const isAssistant = computed(() => props.message.role === 'assistant')

async function copyContent(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.message.content)
    ElMessage.success('已复制回答')
  } catch {
    ElMessage.error('复制失败，请手动选择文本')
  }
}
</script>

<template>
  <div class="message" :class="{ assistant: isAssistant, user: !isAssistant }">
    <div class="avatar" :class="isAssistant ? 'assistant-avatar' : 'user-avatar'">
      {{ isAssistant ? 'K' : '我' }}
    </div>

    <div class="message-body">
      <div class="bubble">
        <MarkdownContent v-if="isAssistant" :content="message.content" />
        <div v-else class="user-text">{{ message.content }}</div>
      </div>

      <SourcePanel v-if="isAssistant && message.sources?.length" :sources="message.sources" />

      <div class="message-meta">
        <el-tag
          v-if="isAssistant && message.sources === null"
          size="small"
          effect="plain"
          type="info"
        >
          未使用知识库
        </el-tag>
        <span>{{ formatTime(message.created_at) }}</span>
        <el-button
          v-if="isAssistant"
          link
          size="small"
          type="primary"
          @click="copyContent"
        >
          复制
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 34px;
  height: 34px;
  flex-shrink: 0;
  border-radius: 10px;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 700;
}

.assistant-avatar {
  color: #fff;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
}

.user-avatar {
  color: #4338ca;
  background: #e8e7fd;
}

.message-body {
  max-width: min(78%, 860px);
  display: flex;
  flex-direction: column;
}

.message.user .message-body {
  align-items: flex-end;
}

.bubble {
  padding: 12px 15px;
  border-radius: 14px;
  background: #fff;
  border: 1px solid var(--ka-border);
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.message.user .bubble {
  background: linear-gradient(135deg, #4f46e5, #6366f1);
  border: none;
  color: #fff;
  border-top-right-radius: 4px;
}

.message.assistant .bubble {
  border-top-left-radius: 4px;
}

.user-text {
  font-size: 14.5px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.message-meta {
  margin-top: 5px;
  padding: 0 3px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11.5px;
  color: #9ca3af;
}

.message.user .message-meta {
  flex-direction: row-reverse;
}
</style>
