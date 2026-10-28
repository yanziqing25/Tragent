import type {
  EChartsOption,
  LineSeriesOption,
  TooltipComponentFormatterCallbackParams,
} from 'echarts'

import type { ReportTradeRow, ResearchReportView, SegmentName } from '@/api/types'
import {
  candidateLabel,
  displayTimestamp,
  formatMoney,
  formatNumber,
  formatPercent,
} from '@/lib/formatters'
import { candidateSegment, orderedCandidates } from '@/lib/report'

export type ChartMetric = 'equity' | 'drawdown'

export interface ChartState {
  visible: string[]
  showAsset: boolean
  showBenchmark: boolean
  metric: ChartMetric
}

export const CHART_PALETTE = {
  light: {
    text: '#09090b',
    muted: '#52525b',
    faint: '#71717a',
    grid: '#f4f4f5',
    axis: '#e4e4e7',
    asset: '#a1a1aa',
    benchmark: '#3f3f46',
    buy: '#16a34a',
    sell: '#dc2626',
    surface: '#ffffff',
    border: '#e4e4e7',
  },
  dark: {
    text: '#fafafa',
    muted: '#a1a1aa',
    faint: '#8b8b93',
    grid: '#1c1c1f',
    axis: '#27272a',
    asset: '#71717a',
    benchmark: '#d4d4d8',
    buy: '#22c55e',
    sell: '#ef4444',
    surface: '#18181b',
    border: '#3f3f46',
  },
}
export type ChartPalette = (typeof CHART_PALETTE)['light']

/** B/S tags sit on the asset line: one strategy, the asset shown, equity view. */
export const tagsVisible = (state: ChartState): boolean =>
  state.visible.length === 1 && state.showAsset && state.metric === 'equity'

// A rounded tag with a stem whose tip marks the traded price.
const TAG_SYMBOL =
  'path://M3,0H17A3,3,0,0,1,20,3V17A3,3,0,0,1,17,20H11V28H9V20H3A3,3,0,0,1,0,17V3A3,3,0,0,1,3,0Z'

interface Point {
  value: number
  trade?: ReportTradeRow
}

export const assetName = (report: ResearchReportView): string =>
  report.bar_interval === '1m'
    ? `${report.symbol} session buy and hold`
    : `${report.symbol} buy and hold`

export const buildSegmentOption = (
  report: ResearchReportView,
  segment: SegmentName,
  state: ChartState,
  colors: Map<string, string>,
  palette: ChartPalette,
): EChartsOption | null => {
  const reference = report.references.find((item) => item.segment === segment)
  if (!reference) return null
  const equity = state.metric === 'equity'
  const timestamps = reference.timestamps.map((value) =>
    displayTimestamp(value, report.bar_interval),
  )
  const candidates = orderedCandidates(report).filter(
    (candidate) => state.visible.includes(candidate.id) && candidateSegment(candidate, segment),
  )
  // Trade tags and tooltip trade details share one rule, so a tag is never left floating.
  const single = tagsVisible(state) ? candidates[0] : undefined
  const singleTrades = single ? (candidateSegment(single, segment)?.trades ?? []) : []
  const tradeAt = new Map(
    singleTrades.map((trade) => [reference.timestamps.indexOf(trade.date), trade]),
  )

  const line = (
    name: string,
    values: number[],
    color: string,
    options: Partial<LineSeriesOption> = {},
  ): LineSeriesOption => ({
    type: 'line',
    name,
    data: values.map((value, index): Point => ({ value, trade: tradeAt.get(index) })),
    showSymbol: false,
    symbol: 'circle',
    symbolSize: 7,
    sampling: 'lttb',
    lineStyle: { color, width: 1.5 },
    itemStyle: { color },
    emphasis: { disabled: true },
    ...options,
  })

  const series: LineSeriesOption[] = []
  if (state.showAsset) {
    const asset = line(
      assetName(report),
      equity ? reference.asset_equity : reference.asset_drawdown,
      palette.asset,
    )
    if (single) {
      asset.markPoint = {
        symbol: TAG_SYMBOL,
        symbolSize: [17, 24],
        symbolOffset: [0, '-50%'],
        silent: true,
        animationDuration: 300,
        label: {
          show: true,
          formatter: '{c}',
          color: '#ffffff',
          fontSize: 10,
          fontWeight: 700,
          offset: [0, -3.5],
        },
        data: singleTrades.flatMap((trade) => {
          const index = reference.timestamps.indexOf(trade.date)
          if (index < 0) return []
          const buy = trade.action === 'buy'
          return [
            {
              name: buy ? 'Buy' : 'Sell',
              coord: [timestamps[index], reference.asset_equity[index]],
              value: buy ? 'B' : 'S',
              itemStyle: { color: buy ? palette.buy : palette.sell },
            },
          ]
        }),
      }
    }
    series.push(asset)
  }
  const benchmark = equity ? reference.benchmark_equity : reference.benchmark_drawdown
  if (state.showBenchmark && benchmark && report.benchmark_symbol) {
    series.push(
      line(report.benchmark_symbol, benchmark, palette.benchmark, {
        lineStyle: { color: palette.benchmark, width: 1.5, type: [5, 4] },
      }),
    )
  }
  for (const candidate of candidates) {
    const values = candidateSegment(candidate, segment)!
    const color = colors.get(candidate.id) ?? palette.text
    series.push(
      line(candidateLabel(candidate), equity ? values.equity : values.drawdown, color, {
        lineStyle: { color, width: candidate.id === report.selected_candidate_id ? 2.25 : 1.75 },
      }),
    )
  }

  const format = (value: number) => (equity ? formatNumber(value, 4) : formatPercent(value))
  return {
    animationDuration: 900,
    animationEasing: 'cubicOut',
    textStyle: { fontFamily: 'Geist Variable, ui-sans-serif, sans-serif' },
    grid: { left: 8, right: 16, top: 28, bottom: 8, containLabel: true },
    xAxis: {
      type: 'category',
      data: timestamps,
      boundaryGap: false,
      axisLine: { lineStyle: { color: palette.axis } },
      axisTick: { show: false },
      axisLabel: { color: palette.faint, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: palette.grid, type: [3, 4] } },
      axisLabel: {
        color: palette.faint,
        formatter: equity
          ? (value: number) => formatNumber(value)
          : (value: number) => formatPercent(value, { digits: 0 }),
      },
    },
    tooltip: {
      trigger: 'axis',
      confine: true,
      backgroundColor: palette.surface,
      borderColor: palette.border,
      padding: [10, 12],
      textStyle: { color: palette.text, fontSize: 12 },
      extraCssText: 'border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.08);',
      axisPointer: { type: 'line', lineStyle: { color: palette.faint, type: [3, 3] } },
      formatter: (raw: TooltipComponentFormatterCallbackParams) =>
        tooltip(raw, format, single ? candidateLabel(single) : null, palette),
    },
    dataZoom: [{ type: 'inside', zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false }],
    series,
  }
}

const tooltip = (
  raw: TooltipComponentFormatterCallbackParams,
  format: (value: number) => string,
  strategy: string | null,
  palette: ChartPalette,
): string => {
  const points = (Array.isArray(raw) ? raw : [raw]).map((item) => ({
    ...item,
    point: item.data as Point,
  }))
  if (!points.length) return ''
  const rows = points.map(
    ({ color, seriesName, point }) =>
      `<div style="display:flex;align-items:center;gap:6px"><span style="width:8px;height:8px;border-radius:2px;background:${String(color)}"></span><span style="flex:1;color:${palette.muted}">${seriesName}</span><b style="font-variant-numeric:tabular-nums;font-weight:500">${format(point.value)}</b></div>`,
  )
  const trade = strategy ? points.find(({ point }) => point.trade)?.point.trade : undefined
  if (trade) {
    const buy = trade.action === 'buy'
    const profit =
      trade.realized_profit === null
        ? ''
        : ` · P/L ${formatMoney(trade.realized_profit, 'USD', true)}`
    rows.push(
      `<div style="margin-top:6px;padding-top:6px;border-top:1px solid ${palette.border}"><b style="color:${buy ? palette.buy : palette.sell}">${buy ? 'Buy' : 'Sell'} · ${strategy}</b><div style="color:${palette.muted};font-variant-numeric:tabular-nums">${formatMoney(trade.price, 'USD')} × ${trade.quantity.toFixed(3)} · cost ${formatMoney(trade.trading_cost, 'USD')} · cash ${formatMoney(trade.cash_balance_after, 'USD')}${profit}</div></div>`,
    )
  }
  return `<div style="min-width:200px;display:flex;flex-direction:column;gap:4px"><b style="font-variant-numeric:tabular-nums">${points[0].name}</b>${rows.join('')}</div>`
}
