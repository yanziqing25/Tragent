<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{ class?: HTMLAttributes['class'] }>()
const model = defineModel<string>({ default: '' })
</script>

<template>
  <textarea
    v-model="model"
    :class="
      cn(
        'w-full resize-y rounded-lg border border-input bg-background px-3 py-2.5 text-sm transition-[border-color,box-shadow] duration-150 placeholder:text-faint focus-visible:border-brand focus-visible:shadow-[0_0_0_4px_color-mix(in_srgb,var(--color-brand)_14%,transparent)] focus-visible:outline-none disabled:opacity-60',
        props.class,
      )
    "
  />
</template>
