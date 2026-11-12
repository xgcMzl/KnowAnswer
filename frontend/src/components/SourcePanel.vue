<script setup lang="ts">
import type { Source } from '@/types'

defineProps<{ sources: Source[] }>()

function scoreText(source: Source): string {
  const score = source.score
  if (score === null || score === undefined || Number.isNaN(score)) return ''
  // RRF 融合分的量级是 0.0x，不能当成百分比展示
  if (source.score_type === 'rrf') return `融合分 ${score.toFixed(4)}`
  return `相似度 ${Math.round(score * 100)}%`
}
</script>

<template>
  <el-collapse v-if="sources.length" class="source-panel">
    <el-collapse-item :title="`引用来源 · ${sources.length} 条`" name="sources">
      <div class="source-list">
        <div
          v-for="source in sources"
          :key="`${source.node_id}-${source.index}`"
          class="source-item"
        >
          <div class="source-head">
            <span class="source-index">[{{ source.index }}]</span>
            <span class="source-name">{{ source.file_name || '未知文件' }}</span>
            <el-tag v-if="source.page_label" size="small" type="info" effect="plain">
              第 {{ source.page_label }} 页
            </el-tag>
            <el-tag v-if="scoreText(source)" size="small" effect="plain">
              {{ scoreText(source) }}
            </el-tag>
          </div>
          <p class="source-text">{{ source.text }}</p>
          <details
            v-if="source.matched_text && source.matched_text !== source.text"
            class="match-block"
          >
            <summary>命中子块</summary>
            <p class="match-text">{{ source.matched_text }}</p>
          </details>
        </div>
      </div>
    </el-collapse-item>
  </el-collapse>
</template>

<style scoped>
.source-panel {
  margin-top: 10px;
  border: 1px solid var(--ka-border);
  border-radius: 9px;
  background: #fafbff;
  overflow: hidden;
}

.source-panel :deep(.el-collapse-item__header) {
  height: 38px;
  padding: 0 12px;
  font-size: 12.5px;
  color: var(--ka-muted);
  background: transparent;
  border: none;
}

.source-panel :deep(.el-collapse-item__wrap) {
  background: transparent;
  border: none;
}

.source-panel :deep(.el-collapse-item__content) {
  padding: 0 12px 10px;
}

.source-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.source-item {
  padding: 9px 10px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #eef0f6;
}

.source-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12.5px;
}

.source-index {
  font-weight: 700;
  color: var(--ka-primary);
}

.source-name {
  font-weight: 600;
  color: #334155;
}

.source-text {
  margin: 6px 0 0;
  font-size: 12.5px;
  line-height: 1.7;
  color: #64748b;
  white-space: pre-wrap;
  max-height: 260px;
  overflow-y: auto;
}

.match-block {
  margin-top: 8px;
  border-top: 1px dashed #eef0f6;
  padding-top: 6px;
}

.match-block summary {
  cursor: pointer;
  font-size: 12px;
  color: var(--ka-muted);
}

.match-text {
  margin: 6px 0 0;
  font-size: 12.5px;
  line-height: 1.7;
  color: #7c8698;
  white-space: pre-wrap;
}
</style>
