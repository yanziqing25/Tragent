import { describe, expect, it, vi } from 'vitest'
import { defineComponent, h, nextTick } from 'vue'
import { createPinia } from 'pinia'
import { PiniaColada } from '@pinia/colada'
import { createApp } from 'vue'

import type { TaskStreamEvent } from '@/api/types'
import { useLiveTask } from '@/queries/research'
import { taskView } from './fixtures'

const stream = vi.hoisted(() => ({
  emit: (() => {}) as (event: TaskStreamEvent) => void,
  finish: () => {},
}))

vi.mock('@/api/client', () => ({
  api: { GET: vi.fn() },
  unwrap: async () => taskView({ run_state: 'queued' }),
  watchTaskEvents: vi.fn(
    (_id: string, _signal: AbortSignal, onEvent: (event: TaskStreamEvent) => void) =>
      new Promise<void>((resolve) => {
        stream.emit = onEvent
        stream.finish = resolve
      }),
  ),
}))

const flush = async () => {
  for (let i = 0; i < 5; i += 1) await nextTick()
  await new Promise((resolve) => setTimeout(resolve))
}

describe('useLiveTask', () => {
  it('follows the stream while the task runs and stops when the result says it is idle', async () => {
    let live!: ReturnType<typeof useLiveTask>
    const app = createApp(
      defineComponent({
        setup() {
          live = useLiveTask('task-1')
          return () => h('div')
        },
      }),
    )
    app.use(createPinia()).use(PiniaColada).mount(document.createElement('div'))
    await flush()

    expect(live.streaming.value).toBe(true)
    stream.emit({
      type: 'agent',
      data: {
        source: 'research_agent',
        kind: 'tool',
        message: 'Research Agent · Running run_backtest',
        tool_name: 'run_backtest',
        agent_status: {
          ...taskView(),
          budget_used: { agent_steps: 4, tool_calls: 3, training_backtests: 2 },
        },
      },
    })
    await flush()
    expect(live.latestMessage.value).toBe('Research Agent · Running run_backtest')
    expect(live.task.data.value?.budget_used.agent_steps).toBe(4)

    stream.emit({ type: 'result', data: taskView({ status: 'selected', run_state: null }) })
    stream.finish()
    await flush()
    expect(live.task.data.value?.status).toBe('selected')
    expect(live.streaming.value).toBe(false)
    app.unmount()
  })
})
