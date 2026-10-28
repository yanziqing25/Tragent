<script setup lang="ts">
import { computed } from 'vue'
import { Check } from '@lucide/vue'

import type { ResearchTaskView } from '@/api/types'

const props = defineProps<{ task: ResearchTaskView }>()

type StepState = 'done' | 'waiting' | 'active' | 'paused' | 'upcoming'

const NOTES: Record<StepState, string> = {
  done: 'Done',
  waiting: 'Waiting for you',
  active: 'In progress',
  paused: 'Paused at a checkpoint',
  upcoming: 'Not started',
}

const steps = computed(() => {
  const task = props.task
  const awaiting =
    task.status === 'awaiting_clarification' || task.status === 'awaiting_confirmation'
  const live: StepState = task.can_resume ? 'paused' : 'active'
  let research: StepState = 'upcoming'
  let critic: StepState = 'upcoming'
  if (task.status === 'running') {
    research = task.active_agent === 'critic' ? 'done' : live
    critic = task.active_agent === 'critic' ? live : 'upcoming'
  } else if (task.last_agent) {
    research = 'done'
    critic = task.critic_round > 0 ? 'done' : 'upcoming'
  }
  return [
    {
      title: 'Goal',
      state: (awaiting ? 'waiting' : 'done') as StepState,
      note: awaiting ? NOTES.waiting : task.status === 'rejected' ? 'Rejected' : 'Confirmed',
    },
    { title: 'Research Agent', state: research, note: NOTES[research] },
    { title: 'Critic Agent', state: critic, note: NOTES[critic] },
  ]
})

const BAR: Record<StepState, string> = {
  done: 'bg-primary',
  waiting: 'animate-breathe bg-warning',
  paused: 'bg-warning',
  active: 'bg-info-soft',
  upcoming: 'bg-border',
}
</script>

<template>
  <ol aria-label="Progress" class="grid grid-cols-3 gap-2">
    <li v-for="(step, index) in steps" :key="step.title" class="flex flex-col gap-1.5">
      <span
        aria-hidden="true"
        class="relative h-[3px] overflow-hidden rounded-full"
        :class="BAR[step.state]"
      >
        <span
          v-if="step.state === 'active'"
          class="absolute inset-y-0 left-0 w-2/5 animate-slide rounded-full bg-brand"
        />
      </span>
      <span
        class="inline-flex items-center gap-1.5 font-medium"
        :class="step.state === 'upcoming' && 'text-faint'"
      >
        <Check v-if="step.state === 'done'" class="size-3.5 text-success" :stroke-width="2.5" />
        {{ index + 1 }} · {{ step.title }}
      </span>
      <span
        class="text-[13px]"
        :class="
          step.state === 'waiting' || step.state === 'paused'
            ? 'text-warning-ink'
            : step.state === 'upcoming'
              ? 'text-faint'
              : 'text-muted-foreground'
        "
      >
        {{ step.note }}
      </span>
    </li>
  </ol>
</template>
