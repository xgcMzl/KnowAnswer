import { request } from './http'
import type { HealthResponse } from '@/types'

export function getHealth(): Promise<HealthResponse> {
  return request('/api/health')
}
