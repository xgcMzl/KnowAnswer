import type { FileStatus } from '@/types'

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** index
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

export function formatTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  const now = new Date()
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  const time = date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
  if (sameDay) return time
  return `${date.getMonth() + 1}-${date.getDate()} ${time}`
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('zh-CN', { hour12: false })
}

export const fileStatusMeta: Record<
  FileStatus,
  { label: string; type: 'info' | 'warning' | 'success' | 'danger' }
> = {
  pending: { label: '排队中', type: 'info' },
  processing: { label: '入库中', type: 'warning' },
  ready: { label: '可检索', type: 'success' },
  failed: { label: '失败', type: 'danger' },
}
