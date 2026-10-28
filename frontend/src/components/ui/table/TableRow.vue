<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{
  class?: HTMLAttributes['class']
}>()
</script>

<template>
  <tr
    :class="
      cn('border-b border-hairline transition-colors duration-150 hover:bg-surface', props.class)
    "
  >
    <slot />
  </tr>
</template>
