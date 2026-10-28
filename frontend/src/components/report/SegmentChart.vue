<script setup lang="ts">
import type { EChartsOption } from 'echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  AxisPointerComponent,
  DataZoomInsideComponent,
  GridComponent,
  MarkPointComponent,
  TooltipComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
  CanvasRenderer,
  LineChart,
  AxisPointerComponent,
  DataZoomInsideComponent,
  GridComponent,
  MarkPointComponent,
  TooltipComponent,
])

defineProps<{ title: string; summary: string; option: EChartsOption }>()
</script>

<template>
  <figure class="flex min-w-0 flex-col gap-2 rounded-xl border border-border p-4">
    <figcaption class="flex justify-between gap-3 text-[13px]">
      <span class="font-medium">{{ title }}</span>
      <span class="text-faint tabular-nums">{{ summary }}</span>
    </figcaption>
    <!-- vue-echarts sizes itself to 100% of its parent, so the parent sets the height. -->
    <div class="h-64">
      <VChart
        :option="option"
        :update-options="{ replaceMerge: ['series'] }"
        autoresize
        :aria-label="`${title} chart`"
      />
    </div>
  </figure>
</template>
