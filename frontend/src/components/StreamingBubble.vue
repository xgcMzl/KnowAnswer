<script setup lang="ts">
import { computed } from 'vue'

import MarkdownContent from './MarkdownContent.vue'
import SourcePanel from './SourcePanel.vue'
import type { Intent, Source } from '@/types'

const props = defineProps<{
  content: string
  sources: Source[]
  intent: Intent | null
}>()

const statusText = computed(() => {
  if (props.intent === 'rag') return '正在结合知识库回答…'
  if (props.intent === 'general') return '直接回答（未使用知识库）…'
  return '正在生成…'
})
</script>

<template>
  <div class="message assistant streaming">
    <div class="avatar assistant-avatar">K</div>
    <div class="message-body">
      <SourcePanel v-if="sources.length" :sources="sources" class="stream-sources" />
      <div class="bubble">
        <div v-if="!content" class="typing">
          <span></span><span></span><span></span>
        </div>
        <template v-else>
          <MarkdownContent :content="content" />
          <span class="cursor"></span>
        </template>
      </div>
      <div class="message-meta">{{ statusText }}</div>
    </div>
  </div>
</template>

<style scoped>
.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
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
  color: #fff;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
}

.message-body {
  max-width: min(78%, 860px);
  display: flex;
  flex-direction: column;
}

.stream-sources {
  margin: 0 0 8px;
}

.bubble {
  padding: 12px 15px;
  border-radius: 14px;
  border-top-left-radius: 4px;
  background: #fff;
  border: 1px solid var(--ka-border);
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.message-meta {
  margin-top: 5px;
  padding: 0 3px;
  font-size: 11.5px;
  color: #9ca3af;
}

.cursor {
  display: inline-block;
  width: 7px;
  height: 15px;
  margin-left: 2px;
  vertical-align: -2px;
  border-radius: 2px;
  background: var(--ka-primary);
  animation: blink 1s steps(1) infinite;
}

.typing {
  display: flex;
  gap: 5px;
  padding: 3px 0;
}

.typing span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #a5b4fc;
  animation: bounce 1.2s infinite ease-in-out;
}

.typing span:nth-child(2) {
  animation-delay: 0.15s;
}

.typing span:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes blink {
  0%,
  50% {
    opacity: 1;
  }
  51%,
  100% {
    opacity: 0;
  }
}

@keyframes bounce {
  0%,
  80%,
  100% {
    transform: translateY(0);
    opacity: 0.5;
  }
  40% {
    transform: translateY(-4px);
    opacity: 1;
  }
}
</style>
