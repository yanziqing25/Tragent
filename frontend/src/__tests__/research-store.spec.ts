import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError, listResearchTasks } from '@/api/api'
import type { AgentEvent } from '@/api/api'
import type { ResearchTaskSummary, ResearchTaskView } from '@/api/generated'
import { useAuthStore } from '@/stores/auth'
import { useResearchStore } from '@/stores/research'

vi.mock('@/api/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/api')>()),
  getResearchTask: vi.fn(),
  listResearchTasks: vi.fn(),
  login: vi.fn(),
  resumeResearchStream: vi.fn(),
  startResearchStream: vi.fn(),
}))

describe('research task state', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.signOut()
    auth.token = 'memory-token'
    vi.clearAllMocks()
  })

  it('tracks offset pagination without inventing a total count', async () => {
    vi.mocked(listResearchTasks).mockResolvedValueOnce(
      Array.from({ length: 20 }, (_, index) => summary(index)),
    )
    const research = useResearchStore()

    await research.loadTasks(20)

    expect(research.offset).toBe(20)
    expect(research.canGoBack).toBe(true)
    expect(research.canGoForward).toBe(true)

    vi.mocked(listResearchTasks).mockResolvedValueOnce([summary(21)])
    await research.loadTasks(40)
    expect(research.canGoForward).toBe(false)
  })

  it('keeps only the latest Agent status from the stream', () => {
    const research = useResearchStore()
    const threadId = '11111111-1111-4111-8111-111111111111'
    const firstAgent: AgentEvent = {
      source: 'goal_parser',
      kind: 'status',
      message: 'Goal Parser · Parsing the research request...',
      tool_name: null,
    }
    const latestAgent: AgentEvent = {
      source: 'critic_agent',
      kind: 'tool',
      message: 'Critic Agent · Running inspect_selected_drawdown...',
      tool_name: 'inspect_selected_drawdown',
      agent_status: {
        active_agent: 'critic',
        budget_used: { agent_steps: 8, tool_calls: 8, training_backtests: 3 },
        budget_remaining: { agent_steps: 3, tool_calls: 3, training_backtests: 0 },
        critic_round: 1,
        critic_verdict: null,
        critic_budget_used: { agent_steps: 2, tool_calls: 1 },
        critic_budget_remaining: { agent_steps: 8, tool_calls: 5 },
      },
    }

    research.applyStreamEvent({ type: 'thread', data: { thread_id: threadId } })
    research.applyStreamEvent({ type: 'result', data: taskView(threadId) })
    research.applyStreamEvent({ type: 'agent', data: firstAgent })
    research.applyStreamEvent({ type: 'agent', data: latestAgent })

    expect(research.currentThreadId).toBe(threadId)
    expect(research.latestAgentMessage).toBe(latestAgent.message)
    expect(research.currentTask?.status).toBe('running')
    expect(research.currentTask?.active_agent).toBe('critic')
    expect(research.currentTask?.budget_used).toEqual(latestAgent.agent_status?.budget_used)
    expect(research.currentTask?.critic_round).toBe(1)
    expect(research.currentTask?.critic_budget_used).toEqual(
      latestAgent.agent_status?.critic_budget_used,
    )
  })

  it('expires the Web session when a protected request returns 401', async () => {
    vi.mocked(listResearchTasks).mockRejectedValueOnce(
      new ApiError(401, 'Incorrect username or password'),
    )
    const auth = useAuthStore()
    const research = useResearchStore()

    await expect(research.loadTasks()).rejects.toBeInstanceOf(ApiError)

    expect(auth.isAuthenticated).toBe(false)
    expect(auth.sessionExpired).toBe(true)
    expect(research.error).toBe('Your session expired. Please sign in again.')
  })
})

const summary = (index: number): ResearchTaskSummary => {
  return {
    thread_id: `11111111-1111-4111-8111-${String(index).padStart(12, '0')}`,
    request_summary: `Research task ${index}`,
    status: 'running',
    updated_at: '2026-08-05T12:00:00Z',
    can_resume: true,
    report_exists: false,
  }
}

const taskView = (threadId: string): ResearchTaskView => {
  return {
    thread_id: threadId,
    status: 'running',
    can_resume: true,
  goal: { objective: 'Research AAPL.' },
  missing_goal_fields: [],
  clarification_hints: [],
  clarification_example: null,
  defaulted_fields: [],
    budget_used: {},
    budget_remaining: {},
    active_agent: 'research',
    critic_round: 0,
    critic_verdict: null,
    critic_budget_used: {},
    critic_budget_remaining: {},
    last_agent: 'research',
    last_action: 'run_backtest',
    last_observation: 'Backtest completed.',
    selected_fingerprint: null,
    stop_reason: null,
    report_exists: false,
    safe_error: null,
  }
}
