import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import * as api from '@/api/knowledge'
import type { KnowledgeBase, KnowledgeFile } from '@/types'

export const useKnowledgeStore = defineStore('knowledge', () => {
  const bases = ref<KnowledgeBase[]>([])
  const activeId = ref<number | null>(null)
  const files = ref<KnowledgeFile[]>([])
  const loadingBases = ref(false)
  const loadingFiles = ref(false)
  const uploading = ref(false)

  let pollTimer: number | null = null

  const activeBase = computed(
    () => bases.value.find((item) => item.id === activeId.value) ?? null,
  )
  const hasProcessingFiles = computed(() =>
    files.value.some(
      (item) => item.status === 'pending' || item.status === 'processing',
    ),
  )

  async function loadBases(): Promise<void> {
    loadingBases.value = true
    try {
      const { items } = await api.listKnowledgeBases()
      bases.value = items
      if (activeId.value && !items.some((item) => item.id === activeId.value)) {
        activeId.value = items[0]?.id ?? null
      }
      if (activeId.value === null && items.length > 0) {
        activeId.value = items[0].id
      }
      if (activeId.value !== null) {
        await loadFiles()
      } else {
        files.value = []
      }
    } finally {
      loadingBases.value = false
    }
  }

  async function selectBase(id: number): Promise<void> {
    activeId.value = id
    await loadFiles()
  }

  async function loadFiles(): Promise<void> {
    if (activeId.value === null) {
      files.value = []
      return
    }
    loadingFiles.value = true
    try {
      const { items } = await api.listFiles(activeId.value)
      files.value = items
      syncPolling()
    } finally {
      loadingFiles.value = false
    }
  }

  async function createBase(name: string, description = ''): Promise<KnowledgeBase> {
    const created = await api.createKnowledgeBase(name, description)
    await loadBases()
    activeId.value = created.id
    await loadFiles()
    return created
  }

  async function renameBase(id: number, name: string): Promise<void> {
    await api.updateKnowledgeBase(id, { name })
    await loadBases()
  }

  async function removeBase(id: number): Promise<void> {
    await api.deleteKnowledgeBase(id)
    if (activeId.value === id) {
      activeId.value = null
    }
    await loadBases()
  }

  async function upload(file: File): Promise<KnowledgeFile> {
    if (activeId.value === null) {
      throw new Error('请先选择知识库')
    }
    uploading.value = true
    try {
      const result = await api.uploadFile(activeId.value, file)
      await loadFiles()
      return result
    } finally {
      uploading.value = false
    }
  }

  async function removeFile(id: number): Promise<void> {
    await api.deleteFile(id)
    await loadFiles()
  }

  function syncPolling(): void {
    if (hasProcessingFiles.value) {
      startPolling()
    } else {
      stopPolling()
    }
  }

  function startPolling(): void {
    if (pollTimer !== null) return
    pollTimer = window.setInterval(async () => {
      try {
        await loadFiles()
      } catch {
        stopPolling()
      }
    }, 2000)
  }

  function stopPolling(): void {
    if (pollTimer !== null) {
      window.clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    bases,
    activeId,
    files,
    loadingBases,
    loadingFiles,
    uploading,
    activeBase,
    hasProcessingFiles,
    loadBases,
    selectBase,
    loadFiles,
    createBase,
    renameBase,
    removeBase,
    upload,
    removeFile,
    stopPolling,
  }
})
