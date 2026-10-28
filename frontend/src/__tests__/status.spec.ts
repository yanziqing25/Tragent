import { describe, expect, it } from 'vitest'

import { taskStatus } from '@/lib/status'

describe('task status presentation', () => {
  it.each([
    [{ status: 'running', run_state: 'running', can_resume: false }, 'Running', 'info', true],
    [{ status: 'running', run_state: 'queued', can_resume: false }, 'Queued', 'neutral', true],
    [{ status: 'running', run_state: null, can_resume: true }, 'Paused', 'warning', false],
    [{ status: 'selected', can_resume: false }, 'Selected', 'success', false],
    [
      { status: 'awaiting_confirmation', can_resume: true },
      'Awaiting confirmation',
      'warning',
      false,
    ],
    [{ status: 'critic_rejected', can_resume: false }, 'Critic rejected', 'danger', false],
  ] as const)('%j reads as %s', (task, label, tone, pulse) => {
    expect(taskStatus(task)).toEqual({ label, tone, pulse })
  })
})
