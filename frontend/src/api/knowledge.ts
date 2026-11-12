import { jsonRequest, request, uploadRequest } from './http'
import type { KnowledgeBase, KnowledgeFile } from '@/types'

export function listKnowledgeBases(): Promise<{ items: KnowledgeBase[] }> {
  return request('/api/knowledge-bases')
}

export function createKnowledgeBase(
  name: string,
  description = '',
): Promise<KnowledgeBase> {
  return jsonRequest('POST', '/api/knowledge-bases', { name, description })
}

export function updateKnowledgeBase(
  id: number,
  payload: { name?: string; description?: string },
): Promise<KnowledgeBase> {
  return jsonRequest('PATCH', `/api/knowledge-bases/${id}`, payload)
}

export function deleteKnowledgeBase(id: number): Promise<void> {
  return jsonRequest('DELETE', `/api/knowledge-bases/${id}`)
}

export function listFiles(knowledgeBaseId: number): Promise<{ items: KnowledgeFile[] }> {
  return request(`/api/knowledge-bases/${knowledgeBaseId}/files`)
}

export function uploadFile(
  knowledgeBaseId: number,
  file: File,
): Promise<KnowledgeFile> {
  return uploadRequest(`/api/knowledge-bases/${knowledgeBaseId}/files`, file)
}

export function getFile(id: number): Promise<KnowledgeFile> {
  return request(`/api/files/${id}`)
}

export function deleteFile(id: number): Promise<void> {
  return jsonRequest('DELETE', `/api/files/${id}`)
}
