import { createParser } from 'eventsource-parser'

import type {
  CheckResult,
  LoginRequest,
  ResearchReportView,
  ResearchTaskSummary,
  ResearchTaskView,
  ResumeResearchRequest,
  StartResearchRequest,
  Token,
} from './generated'

export const ACCESS_TOKEN_KEY = 'tragent.access-token'

export type AgentEvent = {
  /** Project-defined workflow component that emitted this progress event. */
  source: 'goal_parser' | 'workflow' | 'research_agent' | 'critic_agent'
  /** Project-defined semantic category, not a reserved SSE field. */
  kind: 'status' | 'decision' | 'tool' | 'observation'
  /** Bounded progress sentence rendered by the UI. */
  message: string
  /** LangChain Tool involved in tool and observation events. */
  tool_name?: string | null
  /** Optional lightweight counters pushed without refetching the full task. */
  agent_status?: AgentStatusSnapshot | null
}

export type AgentStatusSnapshot = {
  /** Agent expected to make the next model decision. */
  active_agent: 'research' | 'critic' | null
  /** Cumulative Research Agent counters. */
  budget_used: ResearchBudgetSnapshot
  /** Remaining Research Agent allowances. */
  budget_remaining: ResearchBudgetSnapshot
  /** Number of Critic review rounds created so far. */
  critic_round: number
  /** Latest Critic verdict, when submitted. */
  critic_verdict: 'accept' | 'revise' | 'reject' | null
  /** Counters consumed in the active/latest Critic round. */
  critic_budget_used: CriticBudgetSnapshot
  /** Remaining allowances in the active/latest Critic round. */
  critic_budget_remaining: CriticBudgetSnapshot
}

type ResearchBudgetSnapshot = {
  agent_steps: number
  tool_calls: number
  training_backtests: number
}

type CriticBudgetSnapshot = {
  agent_steps: number
  tool_calls: number
}

export type ResearchStreamEvent =
  /** Project event emitted once a new durable thread ID is available. */
  | { type: 'thread'; data: { thread_id: string } }
  /** Project event carrying one LangGraph custom progress update. */
  | { type: 'agent'; data: AgentEvent }
  /** Project terminal event carrying the authoritative task view for this run. */
  | { type: 'result'; data: ResearchTaskView }
  /** Project terminal event used when failure occurs after HTTP streaming has begun. */
  | { type: 'error'; data: { detail: string } }

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export const isUnauthorizedError = (cause: unknown): cause is ApiError => {
  return cause instanceof ApiError && cause.status === 401
}

export const login = async (username: string, password: string): Promise<Token> => {
  const body: LoginRequest = { username, password }
  return requestPublicJson<Token>('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const listResearchTasks = (
  limit = 20,
  offset = 0,
): Promise<ResearchTaskSummary[]> => {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  return requestJson(`/api/research?${query}`)
}

export const getResearchTask = (threadId: string): Promise<ResearchTaskView> => {
  return requestJson(`/api/research/${encodeURIComponent(threadId)}`)
}

export const startResearchStream = (
  body: StartResearchRequest,
  signal: AbortSignal,
  onEvent: (event: ResearchStreamEvent) => void,
): Promise<void> => {
  return streamResearch('/api/research', body, signal, onEvent)
}

export const resumeResearchStream = (
  threadId: string,
  body: ResumeResearchRequest,
  signal: AbortSignal,
  onEvent: (event: ResearchStreamEvent) => void,
): Promise<void> => {
  return streamResearch(
    `/api/research/${encodeURIComponent(threadId)}/resume`,
    body,
    signal,
    onEvent,
  )
}

export const getResearchReport = (
  threadId: string,
  signal?: AbortSignal,
): Promise<ResearchReportView> => {
  return requestJson(`/api/research/${encodeURIComponent(threadId)}/report`, { signal })
}

export const runDoctor = (): Promise<CheckResult[]> => {
  return requestJson('/api/doctor', {
    method: 'POST',
  })
}

const streamResearch = async (
  url: string,
  body: StartResearchRequest | ResumeResearchRequest,
  signal: AbortSignal,
  onEvent: (event: ResearchStreamEvent) => void,
): Promise<void> => {
  const response = await authenticatedFetch(url, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    signal,
  })
  await ensureOk(response)
  if (!response.body) throw new Error('Research stream is unavailable.')

  let terminal = false
  const parser = createParser({
    onEvent(message) {
      const type = message.event as ResearchStreamEvent['type'] | undefined
      if (!type || !['thread', 'agent', 'result', 'error'].includes(type)) {
        throw new Error('Research stream returned an unknown event.')
      }
      const event = { type, data: JSON.parse(message.data) } as ResearchStreamEvent
      terminal ||= type === 'result' || type === 'error'
      onEvent(event)
    },
    onError(error) {
      throw error
    },
  })
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parser.feed(decoder.decode(value, { stream: true }))
  }
  parser.feed(decoder.decode())
  if (!signal.aborted && !terminal) {
    throw new Error('Research stream ended before returning task state.')
  }
}

const authenticatedFetch = (
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> => {
  const token = sessionStorage.getItem(ACCESS_TOKEN_KEY)
  if (!token) throw new Error('Please sign in again.')
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${token}`)
  return fetch(input, { ...init, headers })
}

const requestJson = async <T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> => {
  const response = await authenticatedFetch(input, init)
  await ensureOk(response)
  return response.json() as Promise<T>
}

const requestPublicJson = async <T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> => {
  const response = await fetch(input, init)
  await ensureOk(response)
  return response.json() as Promise<T>
}

const ensureOk = async (response: Response): Promise<void> => {
  if (response.ok) return
  let detail = `Request failed (${response.status}).`
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') detail = body.detail
  } catch {
    // Keep the bounded fallback when the server did not return JSON.
  }
  throw new ApiError(response.status, detail)
}
