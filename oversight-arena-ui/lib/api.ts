/**
 * lib/api.ts
 * All HTTP calls go through the Next.js proxy: /api/* → http://localhost:7860/*
 * The backend (FastAPI) runs on port 7860 (HF Spaces default).
 * Configure BACKEND_URL env var to point to a different backend.
 */

const BASE = '/api'

// ─── Raw backend shapes ────────────────────────────────────────────────────

export interface BackendObservation {
  turn: number
  workers: string[]
  focused_patch_diff: string
  message: string
  reward?: number
  done?: boolean
}

export interface BackendState {
  episode_id: string
  step_count: number
  max_turns: number
  done: boolean
  workers: string[]
  malicious_workers: string[]
  flagged_workers: string[]
  rejected_workers: string[]
  action_history: string[]
  cumulative_reward: number
  terminal_reward: number | null
  difficulty: number
  suspicion_log: Array<{
    turn: number
    action: string
    worker_id: string
    reasoning: string
    cwe_tag: string
    was_malicious: boolean
  }>
  worker_pattern_ids: string[]
  malicious_pattern_ids: string[]
  malicious_tier: string
}

export interface BackendResetResponse {
  observation: BackendObservation
  state?: BackendState
  // The backend may return observation directly at top level
  turn?: number
  workers?: string[]
  focused_patch_diff?: string
  message?: string
}

export interface BackendStepResponse {
  observation?: BackendObservation
  reward: number
  done: boolean
  state?: BackendState
  // top-level fields too
  turn?: number
  focused_patch_diff?: string
}

export interface BackendGraderResponse {
  metric: string
  success: boolean
  reward: number
  f1: number
  precision: number
  recall: number
  tp: number
  fp: number
  fn: number
  malicious_workers: string[]
  flagged_workers: string[]
  step_count: number
  done: boolean
  guardrails_triggered: string[]
  malicious_tier: string
  correct_flags: number
  wrong_flags: number
}

export interface OverseerActionPayload {
  action: 'flag_worker' | 'accept_all' | 'inspect_patch' | 'reject_patch' | 'request_resubmit'
  worker_id?: string
  reasoning?: string
  cwe_tag?: string
  suspicion_scores?: Record<string, number>
}

// ─── Core fetch helpers ────────────────────────────────────────────────────

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`POST ${path} → ${res.status}: ${text}`)
  }
  return res.json()
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`GET ${path} → ${res.status}: ${text}`)
  }
  return res.json()
}

// ─── Public API ────────────────────────────────────────────────────────────

// In lib/api.ts — make sure this is the signature:
export const apiReset = (seed?: number, difficulty?: number) =>
  post<BackendResetResponse>('/reset', {
    ...(seed !== undefined && { seed }),
    ...(difficulty !== undefined && { difficulty }),
  })

export const apiStep = (action: OverseerActionPayload) =>
  post<BackendStepResponse>('/step', action)

export const apiState = () => get<BackendState>('/state')

export const apiGrader = () => get<BackendGraderResponse>('/grader')

export const apiHealth = async (): Promise<boolean> => {
  try {
    await get<{ status: string }>('/health')
    return true
  } catch {
    return false
  }
}

// Legacy hook-style request (used by hooks/useGame.js)
export async function apiRequest(path: string, options?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${options?.method ?? 'GET'} ${path} → ${res.status}: ${text}`)
  }
  return res.json()
}