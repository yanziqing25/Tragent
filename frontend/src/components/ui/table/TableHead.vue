<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{
  class?: HTMLAttributes['class']
}>()
</script>

<template>
  <th
    :class="
      cn(
        'h-10 px-3 text-left align-middle text-[13px] font-medium text-faint first:pl-4 last:pr-4',
        props.class,
      )
    "
  >
    <slot />
  </th>
</template>
