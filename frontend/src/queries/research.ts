import { onScopeDispose, ref, toValue, watch, type MaybeRefOrGetter } from 'vue'
import { useMutation, useQuery, useQueryCache } from '@pinia/colada'

import { api, unwrap, watchTaskEvents } from '@/api/client'
import type { ResearchTaskView, ResumeResearchRequest } from '@/api/types'
import { errorMessage } from '@/lib/errors'

export const PAGE_SIZE = 20

export const researchKeys = {
  list: (offset: number) => ['research', 'list', offset],
  task: (threadId: string) => ['research', 'task', threadId],
  report: (threadId: string) => ['research', 'report', threadId],
}

export const useTaskList = (offset: MaybeRefOrGetter<number>) =>
  useQuery({
    key: () => researchKeys.list(toValue(offset)),
    query: ({ signal }) =>
      unwrap(
        api.GET('/api/research', {
          params: { query: { limit: PAGE_SIZE, offset: toValue(offset) } },
          signal,
        }),
      ),
    // Keep the current page on screen while the next one loads.
    placeholderData: (previous) => previous,
  })

export const useTask = (threadId: MaybeRefOrGetter<string>) =>
  useQuery({
    key: () => researchKeys.task(toValue(threadId)),
    query: ({ signal }) =>
      unwrap(
        api.GET('/api/research/{thread_id}', {
          params: { path: { thread_id: toValue(threadId) } },
          signal,
        }),
      ),
  })

export const useReport = (threadId: MaybeRefOrGetter<string>) =>
  useQuery({
    key: () => researchKeys.report(toValue(threadId)),
    query: ({ signal }) =>
      unwrap(
        api.GET('/api/research/{thread_id}/report', {
          params: { path: { thread_id: toValue(threadId) } },
          signal,
        }),
      ),
  })

export const useStartResearch = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (request: string) => unwrap(api.POST('/api/research', { body: { request } })),
    onSuccess: (task) => {
      cache.setQueryData(researchKeys.task(task.thread_id), task)
      void cache.invalidateQueries({ key: ['research', 'list'] })
    },
  })
}

export const useResumeResearch = (threadId: MaybeRefOrGetter<string>) => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (body: ResumeResearchRequest) =>
      unwrap(
        api.POST('/api/research/{thread_id}/resume', {
          params: { path: { thread_id: toValue(threadId) } },
          body,
        }),
      ),
    onSuccess: (task) => cache.setQueryData(researchKeys.task(task.thread_id), task),
  })
}

/** A task view kept live from its event stream while a worker has it queued or running. */
export const useLiveTask = (threadId: MaybeRefOrGetter<string>) => {
  const cache = useQueryCache()
  const task = useTask(threadId)
  const latestMessage = ref<string | null>(null)
  const streamError = ref<string | null>(null)
  const streaming = ref(false)
  let controller: AbortController | null = null

  const stop = () => {
    controller?.abort()
    controller = null
    streaming.value = false
  }

  const follow = async (id: string) => {
    const current = new AbortController()
    controller = current
    streaming.value = true
    streamError.value = null
    const key = researchKeys.task(id)
    try {
      await watchTaskEvents(id, current.signal, (event) => {
        if (event.type === 'agent') {
          latestMessage.value = event.data.message
          const status = event.data.agent_status
          if (status) {
            cache.setQueryData<ResearchTaskView | undefined>(key, (view) =>
              view ? { ...view, ...status } : view,
            )
          }
        } else if (event.type === 'result') {
          cache.setQueryData(key, event.data)
        } else {
          throw new Error(event.data.detail)
        }
      })
    } catch (cause) {
      if (!current.signal.aborted) streamError.value = errorMessage(cause)
    } finally {
      if (controller === current) {
        controller = null
        streaming.value = false
      }
    }
  }

  watch(
    () => [toValue(threadId), task.data.value?.run_state] as const,
    ([id, runState], previous) => {
      if (previous && previous[0] !== id) {
        stop()
        latestMessage.value = null
      }
      // The server closes the stream once the run ends; a new run opens a new one.
      if (runState && !controller) void follow(id)
    },
    { immediate: true },
  )
  onScopeDispose(stop)

  return { task, latestMessage, streamError, streaming }
}

export const useDiagnostics = () => useMutation({ mutation: () => unwrap(api.POST('/api/doctor')) })
