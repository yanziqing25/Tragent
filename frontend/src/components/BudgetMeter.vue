<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ label: string; used: number; remaining: number; delay?: number }>()

const total = computed(() => props.used + props.remaining)
const width = computed(() => (total.value === 0 ? 0 : (props.used / total.value) * 100))
</script>

<template>
  <div class="flex flex-col gap-1.5">
    <div class="flex justify-between text-[13px]">
      <span class="text-muted-foreground">{{ label }}</span>
      <span class="tabular-nums">{{ used }} / {{ total }}</span>
    </div>
    <div
      role="meter"
      :aria-label="label"
      :aria-valuenow="used"
      :aria-valuemax="total"
      aria-valuemin="0"
      class="h-1.5 overflow-hidden rounded-full bg-muted"
    >
      <div
        class="h-full origin-left animate-grow rounded-full bg-brand transition-[width] duration-500"
        :style="{ width: `${width}%`, animationDelay: `${delay ?? 0}ms` }"
      />
    </div>
  </div>
</template>
