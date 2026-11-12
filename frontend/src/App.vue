<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ChatDotRound, Collection } from '@element-plus/icons-vue'

import { getHealth } from '@/api/health'
import type { HealthResponse } from '@/types'

const health = ref<HealthResponse | null>(null)
const healthy = computed(() => health.value?.status === 'ok')

const healthText = computed(() => {
  if (!health.value) return '无法连接后端服务'
  return [
    `数据库：${health.value.database}`,
    `Milvus：${health.value.milvus.status}`,
    `Embedding：${health.value.embedding_model}`,
    `LLM：${health.value.llm}`,
  ].join('\n')
})

let timer: number | null = null

async function refreshHealth(): Promise<void> {
  try {
    health.value = await getHealth()
  } catch {
    health.value = null
  }
}

onMounted(() => {
  void refreshHealth()
  timer = window.setInterval(() => void refreshHealth(), 30000)
})

onUnmounted(() => {
  if (timer !== null) {
    window.clearInterval(timer)
  }
})
</script>

<template>
  <div class="app-shell">
    <aside class="nav-rail">
      <div class="brand">
        <div class="brand-mark">K</div>
        <div class="brand-text">
          KnowAnswer
          <small>知识库问答</small>
        </div>
      </div>

      <nav>
        <router-link to="/chat" class="nav-item" active-class="active">
          <el-icon :size="18"><ChatDotRound /></el-icon>
          <span>对话</span>
        </router-link>
        <router-link to="/knowledge" class="nav-item" active-class="active">
          <el-icon :size="18"><Collection /></el-icon>
          <span>知识库</span>
        </router-link>
      </nav>

      <el-tooltip placement="right" :show-after="200">
        <template #content>
          <div class="health-tooltip">{{ healthText }}</div>
        </template>
        <div class="nav-footer">
          <span class="health-dot" :class="{ ok: healthy }"></span>
          <span>{{ healthy ? '服务正常' : '服务异常' }}</span>
        </div>
      </el-tooltip>
    </aside>

    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.health-tooltip {
  white-space: pre-line;
  line-height: 1.7;
  font-size: 12px;
}
</style>
