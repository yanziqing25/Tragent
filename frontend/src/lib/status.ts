import type {
  CriticVerdict,
  Order,
  ResearchReportView,
  ResearchTaskView,
  TaskStatus,
} from '@/api/types'

export type Tone = 'neutral' | 'success' | 'info' | 'warning' | 'danger'

export interface StatusPresentation {
  label: string
  tone: Tone
  /** A live state the worker is still changing. */
  pulse: boolean
}

const TASK_STATUS: Record<TaskStatus, Omit<StatusPresentation, 'pulse'>> = {
  awaiting_clarification: { label: 'Needs clarification', tone: 'warning' },
  awaiting_confirmation: { label: 'Awaiting confirmation', tone: 'warning' },
  running: { label: 'Running', tone: 'info' },
  selected: { label: 'Selected', tone: 'success' },
  no_strategy: { label: 'No strategy', tone: 'neutral' },
  critic_rejected: { label: 'Critic rejected', tone: 'danger' },
  failed: { label: 'Failed', tone: 'danger' },
  budget_exhausted: { label: 'Budget exhausted', tone: 'warning' },
  rejected: { label: 'Rejected', tone: 'neutral' },
}

/** A running task is queued, live, or paused at a checkpoint waiting for Resume. */
export const taskStatus = (
  task: Pick<ResearchTaskView, 'status' | 'can_resume'> &
    Partial<Pick<ResearchTaskView, 'run_state'>>,
): StatusPresentation => {
  if (task.status === 'running') {
    if (task.run_state === 'queued') return { label: 'Queued', tone: 'neutral', pulse: true }
    if (task.can_resume) return { label: 'Paused', tone: 'warning', pulse: false }
    return { label: 'Running', tone: 'info', pulse: true }
  }
  return { ...TASK_STATUS[task.status], pulse: false }
}

export const CRITIC_VERDICT: Record<CriticVerdict, { label: string; tone: Tone }> = {
  accept: { label: 'Accept', tone: 'success' },
  revise: { label: 'Revise', tone: 'warning' },
  reject: { label: 'Reject', tone: 'danger' },
}

export const criticReviewLabel = (status: ResearchReportView['critic_review_status']): string =>
  ({
    accepted: 'Critic accepted',
    rejected: 'Critic rejected',
    revision_requested: 'Revision requested',
    incomplete: 'Critic review incomplete',
    not_started: 'Critic review not started',
  })[status ?? 'not_started']

export const ORDER_STATUS: Record<Order['status'], { label: string; tone: Tone }> = {
  pending_submit: { label: 'Sending', tone: 'neutral' },
  submitted: { label: 'Working', tone: 'info' },
  partially_filled: { label: 'Partly filled', tone: 'info' },
  filled: { label: 'Filled', tone: 'success' },
  cancelled: { label: 'Cancelled', tone: 'neutral' },
  rejected: { label: 'Rejected', tone: 'danger' },
}

/** An order the broker may still fill, so it can be cancelled. */
export const isOpenOrder = (order: Pick<Order, 'status'>) =>
  !['filled', 'cancelled', 'rejected'].includes(order.status)
