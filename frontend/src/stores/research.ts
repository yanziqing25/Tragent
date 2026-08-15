import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import {
  getResearchTask,
  listResearchTasks,
  resumeResearchStream,
  startResearchStream,
} from '@/api/api'
import type { ResearchStreamEvent } from '@/api/api'
import type {
  ResearchTaskSummary,
  ResearchTaskView,
  ResumeResearchRequest,
} from '@/api/generated'
import { SESSION_EXPIRED_MESSAGE, useAuthStore } from '@/stores/auth'

const PAGE_SIZE = 20

export const useResearchStore = defineStore('research', () => {
  // Current page of summaries returned by the authenticated list endpoint.
  const tasks = ref<ResearchTaskSummary[]>([])
  // Offset of the currently displayed summary page.
  const offset = ref(0)
  // True while a non-streaming task/list HTTP request is in flight.
  const loading = ref(false)
  // True while one start/resume SSE response is being consumed.
  const streaming = ref(false)
  // Latest bounded API, SSE, or client-side error displayed by the UI.
  const error = ref<string | null>(null)
  // Durable LangGraph thread currently shown by the research detail page.
  const currentThreadId = ref<string | null>(null)
  // Latest authoritative API/SSE view of the current durable task.
  const currentTask = ref<ResearchTaskView | null>(null)
  // Most recent lightweight Agent progress sentence received through SSE.
  const latestAgentMessage = ref<string | null>(null)
  // Browser controller used to cancel the single active SSE fetch stream.
  let controller: AbortController | null = null

  const canGoBack = computed(() => offset.value > 0)
  const canGoForward = computed(() => tasks.value.length === PAGE_SIZE)

  const loadTasks = async (nextOffset = offset.value) => {
    loading.value = true
    error.value = null
    try {
      tasks.value = await listResearchTasks(PAGE_SIZE, nextOffset)
      offset.value = nextOffset
    } catch (cause) {
      error.value = errorMessage(cause)
      throw cause
    } finally {
      loading.value = false
    }
  }

  const loadTask = async (threadId: string) => {
    loading.value = true
    error.value = null
    if (currentThreadId.value !== threadId) latestAgentMessage.value = null
    currentThreadId.value = threadId
    try {
      currentTask.value = await getResearchTask(threadId)
    } catch (cause) {
      error.value = errorMessage(cause)
      throw cause
    } finally {
      loading.value = false
    }
  }

  const startTask = async (request: string): Promise<string> => {
    latestAgentMessage.value = null
    currentTask.value = null
    currentThreadId.value = null
    await runStream((signal) => startResearchStream({ request }, signal, applyStreamEvent))
    if (!currentThreadId.value) throw new Error('Research task ID was not returned.')
    return currentThreadId.value
  }

  const resumeTask = (threadId: string, body: ResumeResearchRequest = {}) => {
    currentThreadId.value = threadId
    latestAgentMessage.value = null
    return runStream((signal) => resumeResearchStream(threadId, body, signal, applyStreamEvent))
  }

  const applyStreamEvent = (event: ResearchStreamEvent) => {
    if (event.type === 'thread') currentThreadId.value = event.data.thread_id
    if (event.type === 'agent') {
      latestAgentMessage.value = event.data.message
      if (currentTask.value && event.data.agent_status) {
        currentTask.value = {
          ...currentTask.value,
          active_agent: event.data.agent_status.active_agent,
          budget_used: event.data.agent_status.budget_used,
          budget_remaining: event.data.agent_status.budget_remaining,
          critic_round: event.data.agent_status.critic_round,
          critic_verdict: event.data.agent_status.critic_verdict,
          critic_budget_used: event.data.agent_status.critic_budget_used,
          critic_budget_remaining: event.data.agent_status.critic_budget_remaining,
        }
      }
    }
    if (event.type === 'result') {
      currentThreadId.value = event.data.thread_id
      currentTask.value = event.data
    }
    if (event.type === 'error') error.value = event.data.detail
  }

  const cancelStream = () => {
    controller?.abort()
  }

  const runStream = async (run: (signal: AbortSignal) => Promise<void>) => {
    cancelStream()
    const activeController = new AbortController()
    controller = activeController
    streaming.value = true
    error.value = null
    try {
      await run(activeController.signal)
      if (error.value) throw new Error(error.value)
    } catch (cause) {
      if (!isAbortError(cause)) error.value = errorMessage(cause)
      throw cause
    } finally {
      if (controller === activeController) {
        controller = null
        streaming.value = false
      }
    }
  }

  return {
    tasks,
    offset,
    loading,
    streaming,
    error,
    currentThreadId,
    currentTask,
    canGoBack,
    canGoForward,
    latestAgentMessage,
    loadTasks,
    loadTask,
    startTask,
    resumeTask,
    applyStreamEvent,
    cancelStream,
  }
})

const errorMessage = (cause: unknown): string => {
  if (useAuthStore().handleUnauthorized(cause)) return SESSION_EXPIRED_MESSAGE
  return cause instanceof Error ? cause.message : 'The request failed.'
}

const isAbortError = (cause: unknown): boolean => {
  return cause instanceof DOMException && cause.name === 'AbortError'
}
