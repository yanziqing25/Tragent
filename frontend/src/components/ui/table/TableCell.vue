<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{
  class?: HTMLAttributes['class']
}>()
</script>

<template>
  <td :class="cn('px-3 py-3 align-middle first:pl-4 last:pr-4', props.class)">
    <slot />
  </td>
</template>
