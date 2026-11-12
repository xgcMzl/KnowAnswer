export const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')

export class ApiError extends Error {
  code: string
  status: number

  constructor(message: string, code = 'UNKNOWN', status = 0) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const data = await response.json()
    const detail = data?.error
    if (detail?.message) {
      return new ApiError(detail.message, detail.code || 'UNKNOWN', response.status)
    }
    if (data?.detail) {
      return new ApiError(String(data.detail), 'INVALID_ARGUMENT', response.status)
    }
  } catch {
    // 非 JSON 响应
  }
  return new ApiError(`请求失败（HTTP ${response.status}）`, 'HTTP_ERROR', response.status)
}

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    throw await toApiError(response)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export function jsonRequest<T>(
  method: 'POST' | 'PATCH' | 'PUT' | 'DELETE',
  path: string,
  body?: unknown,
): Promise<T> {
  return request<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

export function uploadRequest<T>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  return request<T>(path, { method: 'POST', body: form })
}
