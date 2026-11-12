<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, EditPen, Plus } from '@element-plus/icons-vue'

import { useChatStore } from '@/stores/chat'
import { formatTime } from '@/utils/format'
import type { Conversation } from '@/types'

const emit = defineEmits<{ (event: 'new'): void }>()

const chat = useChatStore()

async function selectConversation(conversation: Conversation): Promise<void> {
  if (chat.streaming) return
  try {
    await chat.activate(conversation.id)
  } catch (error) {
    ElMessage.error((error as Error).message || '加载会话失败')
  }
}

async function renameConversation(conversation: Conversation): Promise<void> {
  try {
    const { value } = await ElMessageBox.prompt('请输入新的会话标题', '重命名会话', {
      inputValue: conversation.title || '未命名对话',
      confirmButtonText: '保存',
      cancelButtonText: '取消',
      inputValidator: (input: string) => (input.trim() ? true : '标题不能为空'),
    })
    if (value?.trim()) {
      await chat.renameConversation(conversation.id, value.trim())
      ElMessage.success('已重命名')
    }
  } catch (error) {
    if (error instanceof Error) ElMessage.error(error.message)
  }
}

async function removeConversation(conversation: Conversation): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `确定删除会话“${conversation.title || '未命名对话'}”？消息记录会被一并删除。`,
      '删除会话',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
    await chat.removeConversation(conversation.id)
    ElMessage.success('会话已删除')
  } catch (error) {
    if (error instanceof Error) ElMessage.error(error.message)
  }
}
</script>

<template>
  <aside class="conversation-sidebar">
    <div class="sidebar-header">
      <span>对话</span>
      <el-button circle size="small" :icon="Plus" @click="emit('new')" />
    </div>

    <el-scrollbar class="conversation-scroll">
      <div
        v-for="conversation in chat.conversations"
        :key="conversation.id"
        class="conversation-item"
        :class="{ active: conversation.id === chat.activeId }"
        @click="selectConversation(conversation)"
      >
        <div class="conversation-main">
          <div class="conversation-title">
            {{ conversation.title || '未命名对话' }}
          </div>
          <div class="conversation-time">{{ formatTime(conversation.updated_at) }}</div>
        </div>
        <div class="conversation-actions" @click.stop>
          <el-button
            link
            size="small"
            :icon="EditPen"
            @click="renameConversation(conversation)"
          />
          <el-button
            link
            size="small"
            type="danger"
            :icon="Delete"
            @click="removeConversation(conversation)"
          />
        </div>
      </div>

      <el-empty
        v-if="!chat.conversations.length && !chat.loadingConversations"
        description="还没有对话"
        :image-size="70"
      />
    </el-scrollbar>
  </aside>
</template>

<style scoped>
.conversation-sidebar {
  width: 270px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #fbfcfe;
  border-right: 1px solid var(--ka-border);
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 16px 12px;
  font-size: 14px;
  font-weight: 600;
  color: #334155;
}

.conversation-scroll {
  flex: 1;
  min-height: 0;
  padding: 0 10px 16px;
}

.conversation-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 11px;
  margin-bottom: 4px;
  border-radius: 9px;
  cursor: pointer;
  transition: background 0.15s ease;
}

.conversation-item:hover {
  background: #eef2ff;
}

.conversation-item.active {
  background: #e8e7fd;
}

.conversation-main {
  flex: 1;
  min-width: 0;
}

.conversation-title {
  font-size: 13.5px;
  color: #1f2937;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conversation-item.active .conversation-title {
  color: #4338ca;
  font-weight: 600;
}

.conversation-time {
  margin-top: 3px;
  font-size: 11px;
  color: #9ca3af;
}

.conversation-actions {
  display: flex;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.conversation-item:hover .conversation-actions {
  opacity: 1;
}
</style>
