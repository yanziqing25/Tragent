<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import Plotly from 'plotly.js-basic-dist-min'
import type { Data, Layout, LegendClickEvent, PlotlyHTMLElement } from 'plotly.js'

import type { ResearchReportView } from '@/api/generated'
import { eligibleTradeGroup, tradeDetailKey } from '@/components/reportChart'
import { buildReportFigure } from '@/components/reportFigure'

const props = defineProps<{
  report: ResearchReportView
}>()
const emit = defineEmits<{ focusTradeDetails: [candidateKey: string] }>()

const root = ref<HTMLDivElement | null>(null)
let chart: PlotlyHTMLElement | null = null
let appliedTradeGroup: string | null | undefined
let synchronizing = false
let syncQueued = false
const tradeHoverData = new Map<number, unknown[]>()
const blankTradeHoverData = new Map<number, string[]>()

type ChartTrace = Data & {
  customdata?: unknown[]
  legendgroup?: string
}

onMounted(async () => {
  if (!root.value) return
  const figure = buildReportFigure(props.report)
  const renderedChart = await Plotly.newPlot(
    root.value,
    figure.data,
    figure.layout,
    figure.config,
  )
  chart = renderedChart

  const traces = renderedChart.data as ChartTrace[]
  traces.forEach((trace, index) => {
    if (!tradeDetailKey(trace) || !Array.isArray(trace.customdata)) return
    tradeHoverData.set(index, trace.customdata)
    blankTradeHoverData.set(index, trace.customdata.map(() => ''))
  })
  renderedChart.on('plotly_legendclick', focusTradeDetails)
  renderedChart.on('plotly_legenddoubleclick', focusTradeDetails)
  renderedChart.on('plotly_restyle', syncTradeOverlay)
  syncTradeOverlay()
})

onUnmounted(() => {
  if (chart) Plotly.purge(chart)
  chart = null
})

const focusTradeDetails = (event: LegendClickEvent): boolean => {
  const key = tradeDetailKey(event.data[event.curveNumber])
  if (key) emit('focusTradeDetails', key)
  return true
}

const syncTradeOverlay = () => {
  if (!chart) return
  if (synchronizing) {
    syncQueued = true
    return
  }

  const visibleGroup = eligibleTradeGroup(chart.data)
  if (visibleGroup === appliedTradeGroup) return
  appliedTradeGroup = visibleGroup

  const shapeUpdates: Record<string, boolean> = {}
  const shapes = chart.layout.shapes ?? []
  shapes.forEach((shape, index) => {
    const group = 'legendgroup' in shape ? shape.legendgroup : undefined
    if (typeof group !== 'string') return
    const visible = group === visibleGroup
    if (shape.visible !== visible) shapeUpdates[`shapes[${index}].visible`] = visible
  })
  const traceIndices = [...tradeHoverData.keys()]
  const customdata = traceIndices.map((index) => {
    const trace = chart!.data[index] as ChartTrace | undefined
    const group = trace && 'legendgroup' in trace ? trace.legendgroup : undefined
    return group === visibleGroup ? tradeHoverData.get(index) : blankTradeHoverData.get(index)
  })
  if (traceIndices.length === 0 && Object.keys(shapeUpdates).length === 0) return

  synchronizing = true
  const updates: Array<Promise<PlotlyHTMLElement>> = []
  if (traceIndices.length) {
    updates.push(Plotly.restyle(chart, { customdata } as Data, traceIndices))
  }
  if (Object.keys(shapeUpdates).length) {
    updates.push(Plotly.relayout(chart, shapeUpdates as Partial<Layout>))
  }
  Promise.all(updates).finally(() => {
    synchronizing = false
    if (!syncQueued) return
    syncQueued = false
    syncTradeOverlay()
  })
}
</script>

<template>
  <div ref="root" class="report-chart" aria-label="Equity and drawdown chart" />
</template>

<style scoped>
.report-chart {
  width: 100%;
  min-height: 75rem;
}
</style>
