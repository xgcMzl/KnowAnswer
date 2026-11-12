<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { UploadRequestOptions } from 'element-plus'
import {
  Delete,
  Document,
  EditPen,
  Plus,
  UploadFilled,
} from '@element-plus/icons-vue'

import { useKnowledgeStore } from '@/stores/knowledge'
import { fileStatusMeta, formatBytes, formatDateTime } from '@/utils/format'
import type { FileStatus, KnowledgeBase, KnowledgeFile } from '@/types'

function statusMeta(status: string) {
  return fileStatusMeta[status as FileStatus] ?? { label: status, type: 'info' as const }
}

const knowledge = useKnowledgeStore()

const createVisible = ref(false)
const newName = ref('')
const newDescription = ref('')
const creating = ref(false)

onMounted(async () => {
  try {
    await knowledge.loadBases()
  } catch (error) {
    ElMessage.error((error as Error).message || '加载知识库失败')
  }
})

onUnmounted(() => {
  knowledge.stopPolling()
})

async function selectBase(base: KnowledgeBase): Promise<void> {
  try {
    await knowledge.selectBase(base.id)
  } catch (error) {
    ElMessage.error((error as Error).message || '加载文件列表失败')
  }
}

async function submitCreate(): Promise<void> {
  const name = newName.value.trim()
  if (!name) {
    ElMessage.warning('请输入知识库名称')
    return
  }
  creating.value = true
  try {
    await knowledge.createBase(name, newDescription.value.trim())
    ElMessage.success('知识库已创建')
    createVisible.value = false
    newName.value = ''
    newDescription.value = ''
  } catch (error) {
    ElMessage.error((error as Error).message || '创建失败')
  } finally {
    creating.value = false
  }
}

async function renameBase(base: KnowledgeBase): Promise<void> {
  try {
    const { value } = await ElMessageBox.prompt('请输入新的知识库名称', '重命名知识库', {
      inputValue: base.name,
      confirmButtonText: '保存',
      cancelButtonText: '取消',
      inputValidator: (input: string) => (input.trim() ? true : '名称不能为空'),
    })
    if (value?.trim()) {
      await knowledge.renameBase(base.id, value.trim())
      ElMessage.success('已重命名')
    }
  } catch (error) {
    if (error instanceof Error) ElMessage.error(error.message)
  }
}

async function removeBase(base: KnowledgeBase): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `删除知识库“${base.name}”？其中的文件和向量都会被删除。`,
      '删除知识库',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
    await knowledge.removeBase(base.id)
    ElMessage.success('知识库已删除')
  } catch (error) {
    if (error instanceof Error) ElMessage.error(error.message)
  }
}

async function handleUpload(options: UploadRequestOptions): Promise<void> {
  try {
    await knowledge.upload(options.file)
    ElMessage.success(`已上传 ${options.file.name}，正在后台入库`)
  } catch (error) {
    ElMessage.error((error as Error).message || '上传失败')
    throw error
  }
}

async function removeFile(row: KnowledgeFile): Promise<void> {
  try {
    await ElMessageBox.confirm(`确定删除文件“${row.file_name}”？`, '删除文件', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await knowledge.removeFile(row.id)
    ElMessage.success('文件已删除')
  } catch (error) {
    if (error instanceof Error) ElMessage.error(error.message)
  }
}

function onRemoveFile(row: unknown): void {
  void removeFile(row as KnowledgeFile)
}
</script>

<template>
  <div class="knowledge-view">
    <aside class="kb-sidebar">
      <div class="kb-sidebar-header">
        <span>知识库</span>
        <el-button circle size="small" :icon="Plus" @click="createVisible = true" />
      </div>

      <el-scrollbar class="kb-scroll">
        <div
          v-for="base in knowledge.bases"
          :key="base.id"
          class="kb-item"
          :class="{ active: base.id === knowledge.activeId }"
          @click="selectBase(base)"
        >
          <div class="kb-item-main">
            <div class="kb-name">{{ base.name }}</div>
            <div class="kb-meta">{{ base.file_count }} 个文件</div>
          </div>
          <div class="kb-actions" @click.stop>
            <el-button link size="small" :icon="EditPen" @click="renameBase(base)" />
            <el-button
              link
              size="small"
              type="danger"
              :icon="Delete"
              @click="removeBase(base)"
            />
          </div>
        </div>

        <el-empty
          v-if="!knowledge.bases.length && !knowledge.loadingBases"
          description="还没有知识库"
          :image-size="70"
        />
      </el-scrollbar>
    </aside>

    <section class="kb-main">
      <template v-if="knowledge.activeBase">
        <header class="kb-header">
          <div>
            <h2>{{ knowledge.activeBase.name }}</h2>
            <p>{{ knowledge.activeBase.description || '暂无描述' }}</p>
          </div>
          <el-tag effect="plain">{{ knowledge.files.length }} 个文件</el-tag>
        </header>

        <div class="upload-zone">
          <el-upload
            drag
            multiple
            :show-file-list="false"
            :http-request="handleUpload"
            :disabled="knowledge.uploading"
            accept=".pdf,.docx,.md,.txt"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="upload-title">拖拽文件到此处，或点击上传</div>
            <div class="upload-tip">
              支持 PDF / DOCX / Markdown / TXT，单文件不超过 20MB
            </div>
          </el-upload>
        </div>

        <el-table
          v-loading="knowledge.loadingFiles"
          :data="knowledge.files"
          class="file-table"
          empty-text="还没有文件，上传一个开始吧"
        >
          <el-table-column prop="file_name" label="文件名" min-width="220">
            <template #default="{ row }">
              <div class="file-name">
                <el-icon><Document /></el-icon>
                <span>{{ row.file_name }}</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="大小" width="100">
            <template #default="{ row }">{{ formatBytes(row.file_size) }}</template>
          </el-table-column>
          <el-table-column label="状态" width="120">
            <template #default="{ row }">
              <el-tooltip
                :disabled="row.status !== 'failed' || !row.error_message"
                :content="row.error_message || ''"
                placement="top"
              >
                <el-tag :type="statusMeta(row.status).type" size="small">
                  {{ statusMeta(row.status).label }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column prop="chunk_count" label="切片" width="80" />
          <el-table-column label="更新时间" width="170">
            <template #default="{ row }">{{ formatDateTime(row.updated_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="90" fixed="right">
            <template #default="{ row }">
              <el-button
                link
                type="danger"
                :disabled="row.status === 'processing'"
                @click="onRemoveFile(row)"
              >
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </template>

      <el-empty v-else description="请选择或创建一个知识库">
        <el-button type="primary" @click="createVisible = true">新建知识库</el-button>
      </el-empty>
    </section>

    <el-dialog v-model="createVisible" title="新建知识库" width="420px">
      <el-form label-position="top">
        <el-form-item label="名称">
          <el-input
            v-model="newName"
            maxlength="128"
            show-word-limit
            placeholder="例如：产品手册"
          />
        </el-form-item>
        <el-form-item label="描述">
          <el-input
            v-model="newDescription"
            type="textarea"
            :rows="3"
            placeholder="可选"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="submitCreate">
          创建
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.knowledge-view {
  display: flex;
  height: 100%;
  overflow: hidden;
}

.kb-sidebar {
  width: 270px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #fbfcfe;
  border-right: 1px solid var(--ka-border);
}

.kb-sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 16px 12px;
  font-size: 14px;
  font-weight: 600;
  color: #334155;
}

.kb-scroll {
  flex: 1;
  min-height: 0;
  padding: 0 10px 16px;
}

.kb-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 11px;
  margin-bottom: 4px;
  border-radius: 9px;
  cursor: pointer;
  transition: background 0.15s ease;
}

.kb-item:hover {
  background: #eef2ff;
}

.kb-item.active {
  background: #e8e7fd;
}

.kb-item-main {
  flex: 1;
  min-width: 0;
}

.kb-name {
  font-size: 13.5px;
  color: #1f2937;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.kb-item.active .kb-name {
  color: #4338ca;
  font-weight: 600;
}

.kb-meta {
  margin-top: 3px;
  font-size: 11px;
  color: #9ca3af;
}

.kb-actions {
  display: flex;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.kb-item:hover .kb-actions {
  opacity: 1;
}

.kb-main {
  flex: 1;
  min-width: 0;
  overflow-y: auto;
  padding: 24px 30px 40px;
}

.kb-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
}

.kb-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 650;
  color: #0f172a;
}

.kb-header p {
  margin: 6px 0 0;
  font-size: 13px;
  color: var(--ka-muted);
}

.upload-zone {
  margin-bottom: 20px;
}

.upload-zone :deep(.el-upload-dragger) {
  padding: 22px 16px;
  border-radius: 12px;
  border: 1px dashed #c7cbe8;
  background: #fafbff;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.upload-zone :deep(.el-upload-dragger:hover) {
  border-color: var(--ka-primary);
  background: #f5f6ff;
}

.upload-icon {
  font-size: 30px;
  color: var(--ka-primary);
}

.upload-title {
  margin-top: 8px;
  font-size: 13.5px;
  color: #334155;
}

.upload-tip {
  margin-top: 5px;
  font-size: 11.5px;
  color: #9ca3af;
}

.file-table {
  border-radius: 10px;
  overflow: hidden;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #334155;
}

@media (max-width: 900px) {
  .kb-main {
    padding: 18px 16px 30px;
  }
}
</style>
