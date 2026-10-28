<script setup lang="ts">
import { computed } from 'vue'

import type { ResearchTaskView } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { taskStatus } from '@/lib/status'

const props = defineProps<{
  task: Pick<ResearchTaskView, 'status' | 'can_resume'> &
    Partial<Pick<ResearchTaskView, 'run_state'>>
  size?: 'default' | 'sm'
}>()

const state = computed(() => taskStatus(props.task))
const DOT = {
  neutral: 'bg-faint',
  success: 'bg-success',
  info: 'bg-brand',
  warning: 'bg-warning',
  danger: 'bg-destructive',
} as const
</script>

<template>
  <Badge :variant="state.tone" :size="size">
    <span
      aria-hidden="true"
      class="size-1.5 rounded-full"
      :class="[DOT[state.tone], state.pulse && 'animate-breathe']"
    />
    {{ state.label }}
  </Badge>
</template>
