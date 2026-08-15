import type { Config, Data, Layout } from 'plotly.js'

import type {
  ReportCandidate,
  ReportCandidateSegment,
  ReportReferenceSegment,
  ResearchReportView,
  StrategyName,
} from '@/api/generated'

const STRATEGY_ORDER: StrategyName[] = ['moving_average', 'rsi', 'macd']
const STRATEGY_COLORS: Record<StrategyName, string> = {
  moving_average: '#2563eb',
  rsi: '#16a34a',
  macd: '#f97316',
}
const PLOTLY_DASHES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'] as const
const TRADE_COLORS = { buy: '#16a34a', sell: '#dc2626' } as const
const TRADE_TAG_PATH = 'M 0,0 L 0,8 M 0,8 L 11,18 L 11,42 L -11,42 L -11,18 Z'

type ReportFigure = {
  data: Data[]
  layout: Partial<Layout>
  config: Partial<Config>
}

export const candidateLabel = (candidate: ReportCandidate): string => {
  const parameters = Object.entries(candidate.parameters)
    .map(([name, value]) => `${name}=${value}`)
    .join(', ')
  return `${candidate.strategy_name} (${parameters})`
}

export const buildReportFigure = (report: ResearchReportView): ReportFigure => {
  const candidates = orderedFinalCandidates(report.candidates)
  const defaultCandidateId = defaultReportCandidateId(report)
  const variantIndices = candidateVariantIndices(candidates)
  const data: Data[] = []
  const shapes: NonNullable<Layout['shapes']> = []
  const annotations: NonNullable<Layout['annotations']> = []
  const layout = baseLayout(shapes, annotations)

  const plotRows = [
    {
      curve: 'equity' as const,
      assetCurve: 'asset_equity' as const,
      benchmarkCurve: 'benchmark_equity' as const,
      title: 'Equity',
      yTitle: 'Growth of 1.0',
      valueFormat: '.4f',
    },
    {
      curve: 'drawdown' as const,
      assetCurve: 'asset_drawdown' as const,
      benchmarkCurve: 'benchmark_drawdown' as const,
      title: 'Drawdown',
      yTitle: 'Drawdown',
      valueFormat: '.2%',
      tickformat: '.0%',
    },
  ]

  plotRows.forEach((plotRow, rowIndex) => {
    ;(['training', 'test'] as const).forEach((segmentName, columnIndex) => {
      const reference = report.references.find((item) => item.segment === segmentName)
      if (!reference) return
      const axisNumber = rowIndex * 2 + columnIndex + 1
      const xaxis = axisName('x', axisNumber)
      const yaxis = axisName('y', axisNumber)
      const showLegend = rowIndex === 0 && columnIndex === 0
      const timestamps = reference.timestamps.map((value) => displayTimestamp(value, report.bar_interval))
      const timeFormat = report.bar_interval === '1m' ? '%Y-%m-%d %H:%M' : '%Y-%m-%d'
      const assetName =
        report.bar_interval === '1m' ? `${report.symbol} Session B&H` : `${report.symbol} B&H`

      data.push(
        lineTrace({
          x: timestamps,
          y: reference[plotRow.assetCurve],
          name: assetName,
          legendgroup: assetName,
          xaxis,
          yaxis,
          showlegend: showLegend,
          color: '#64748b',
          dash: 'dot',
          hovertemplate: `%{x|${timeFormat}}<br>%{y:${plotRow.valueFormat}}<extra>${assetName}</extra>`,
          meta: { reference_role: 'asset_buy_and_hold' },
        }),
      )

      const benchmark = reference[plotRow.benchmarkCurve]
      if (benchmark && report.benchmark_symbol) {
        const benchmarkName = `${report.benchmark_symbol} B&H`
        data.push(
          lineTrace({
            x: timestamps,
            y: benchmark,
            name: benchmarkName,
            legendgroup: benchmarkName,
            xaxis,
            yaxis,
            showlegend: showLegend,
            color: '#0f172a',
            dash: 'dash',
            hovertemplate: `%{x|${timeFormat}}<br>%{y:${plotRow.valueFormat}}<extra>${benchmarkName}</extra>`,
            meta: { reference_role: 'benchmark_buy_and_hold' },
          }),
        )
      }

      candidates.forEach((candidate) => {
        const segment = candidate.segments.find((item) => item.segment === segmentName)!
        const label = candidateLabel(candidate)
        const visible = candidate.id === defaultCandidateId
        const customdata =
          plotRow.curve === 'equity'
            ? tradeHoverData(reference, segment)
            : undefined
        const hovertemplate = customdata
          ? `%{x|${timeFormat}}<br>Equity %{y:.4f}%{customdata}<extra>${label}</extra>`
          : `%{x|${timeFormat}}<br>%{y:${plotRow.valueFormat}}<extra>${label}</extra>`

        data.push(
          lineTrace({
            x: timestamps,
            y: segment[plotRow.curve],
            name: candidateLegendLabel(candidate),
            legendgroup: candidate.id,
            xaxis,
            yaxis,
            showlegend: showLegend,
            visible: visible ? true : 'legendonly',
            color: STRATEGY_COLORS[candidate.strategy_name],
            dash: PLOTLY_DASHES[variantIndices.get(candidate.id)! % PLOTLY_DASHES.length]!,
            width: candidate.id === report.selected_candidate_id ? 3 : 2,
            hovertemplate,
            customdata,
            meta: { trade_detail_key: candidate.id },
          }),
        )

        if (plotRow.curve === 'equity') {
          shapes.push(...tradeShapes(reference, segment, candidate.id, visible, xaxis, yaxis, report))
        }
      })

      setAxes(layout, axisNumber, rowIndex, columnIndex, report.bar_interval, plotRow.yTitle, plotRow.tickformat)
      annotations.push({
        text: `${plotRow.title} · ${segmentName === 'training' ? 'In-sample' : 'Out-of-sample'}`,
        x: columnIndex === 0 ? 0.23 : 0.77,
        y: rowIndex === 0 ? 1.04 : 0.46,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { size: 16 },
      })
    })
  })

  return {
    data,
    layout,
    config: {
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      toImageButtonOptions: {
        format: 'png',
        filename: 'research-curves',
        scale: 2,
      },
    },
  }
}

export const defaultReportCandidateId = (report: ResearchReportView): string | null => {
  return report.selected_candidate_id
    ?? orderedFinalCandidates(report.candidates)[0]?.id
    ?? report.candidates[0]?.id
    ?? null
}

const orderedFinalCandidates = (candidates: ReportCandidate[]): ReportCandidate[] => {
  return candidates
    .filter((candidate) => candidate.segments.some((segment) => segment.segment === 'test'))
    .sort(
      (left, right) =>
        STRATEGY_ORDER.indexOf(left.strategy_name) - STRATEGY_ORDER.indexOf(right.strategy_name),
    )
}

const candidateVariantIndices = (candidates: ReportCandidate[]): Map<string, number> => {
  const counts = new Map<StrategyName, number>()
  return new Map(
    candidates.map((candidate) => {
      const index = counts.get(candidate.strategy_name) ?? 0
      counts.set(candidate.strategy_name, index + 1)
      return [candidate.id, index]
    }),
  )
}

const candidateLegendLabel = (candidate: ReportCandidate): string => {
  const values = candidate.parameters
  if (candidate.strategy_name === 'moving_average') {
    return `MA · ${values.short_window}/${values.long_window}`
  }
  if (candidate.strategy_name === 'rsi') {
    return `RSI · ${values.rsi_window} · ${values.entry_threshold}/${values.exit_threshold}`
  }
  return `MACD · ${values.fast_period}/${values.slow_period}/${values.signal_period}`
}

type LineTraceInput = {
  x: string[]
  y: number[]
  name: string
  legendgroup: string
  xaxis: string
  yaxis: string
  showlegend: boolean
  color: string
  dash: string
  hovertemplate: string
  visible?: true | 'legendonly'
  width?: number
  customdata?: string[]
  meta: Record<string, string>
}

const lineTrace = (input: LineTraceInput): Data => {
  const { color, dash, width, ...trace } = input
  return {
    type: 'scatter',
    mode: 'lines',
    ...trace,
    line: { color, dash, width: width ?? 2 },
  } as Data
}

const tradeHoverData = (
  reference: ReportReferenceSegment,
  segment: ReportCandidateSegment,
): string[] => {
  const trades = new Map(
    segment.trades.map((trade) => [
      trade.date,
      `<br><b>${trade.action === 'buy' ? 'Buy' : 'Sell'}</b> · Price $${money(trade.price)}`
        + ` · Quantity ${trade.quantity.toFixed(4)}`
        + ` · Cost $${money(trade.trading_cost)}`
        + ` · Cash $${money(trade.cash_balance_after)}`,
    ]),
  )
  return reference.timestamps.map((timestamp) => trades.get(timestamp) ?? '')
}

const tradeShapes = (
  reference: ReportReferenceSegment,
  segment: ReportCandidateSegment,
  candidateId: string,
  visible: boolean,
  xref: string,
  yref: string,
  report: ResearchReportView,
): NonNullable<Layout['shapes']> => {
  return segment.trades.flatMap((trade) => {
    const index = reference.timestamps.indexOf(trade.date)
    if (index < 0) return []
    const label = trade.action === 'buy' ? 'B' : 'S'
    const color = TRADE_COLORS[trade.action]
    return [{
      type: 'path',
      path: TRADE_TAG_PATH,
      xsizemode: 'pixel',
      ysizemode: 'pixel',
      xanchor: displayTimestamp(trade.date, report.bar_interval),
      yanchor: reference.asset_equity[index],
      xref,
      yref,
      fillcolor: color,
      line: { color, width: 1 },
      layer: 'above',
      legendgroup: candidateId,
      visible,
      showlegend: false,
      label: {
        text: label,
        textposition: 'top center',
        yanchor: 'top',
        padding: 7,
        font: { color: 'white', family: 'Arial Black, sans-serif', size: 11 },
      },
    }]
  }) as NonNullable<Layout['shapes']>
}

const baseLayout = (
  shapes: NonNullable<Layout['shapes']>,
  annotations: NonNullable<Layout['annotations']>,
): Partial<Layout> => {
  const layout = {
    autosize: true,
    dragmode: 'zoom',
    hovermode: 'x unified',
    template: 'plotly_white',
    height: 1200,
    margin: { l: 56, r: 24, t: 62, b: 150 },
    legend: {
      orientation: 'h',
      groupclick: 'togglegroup',
      entrywidth: 280,
      entrywidthmode: 'pixels',
      yanchor: 'top',
      y: -0.18,
      xanchor: 'left',
      x: 0,
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#ffffff',
    shapes,
    annotations,
  }
  return layout as unknown as Partial<Layout>
}

const setAxes = (
  layout: Partial<Layout>,
  axisNumber: number,
  rowIndex: number,
  columnIndex: number,
  barInterval: ResearchReportView['bar_interval'],
  yTitle: string,
  tickformat?: string,
) => {
  const xKey = axisName('xaxis', axisNumber) as keyof Layout
  const yKey = axisName('yaxis', axisNumber) as keyof Layout
  const xaxis = axisName('x', axisNumber)
  const yaxis = axisName('y', axisNumber)
  const xDomain = columnIndex === 0 ? [0, 0.46] : [0.54, 1]
  const yDomain = rowIndex === 0 ? [0.58, 1] : [0, 0.42]
  const minuteRangebreaks =
    barInterval === '1m'
      ? [{ bounds: ['sat', 'mon'] }, { bounds: [16, 9.5], pattern: 'hour' }]
      : undefined
  const axes = layout as Record<string, unknown>
  axes[xKey] = {
    anchor: yaxis,
    domain: xDomain,
    title: { text: barInterval === '1m' ? 'US/Eastern time' : 'Date' },
    rangebreaks: minuteRangebreaks,
    showgrid: true,
    gridcolor: '#e7edf5',
  }
  axes[yKey] = {
    anchor: xaxis,
    domain: yDomain,
    title: { text: columnIndex === 0 ? yTitle : undefined },
    tickformat,
    showgrid: true,
    gridcolor: '#e7edf5',
  }
}

const axisName = (axis: string, number: number): string => {
  return number === 1 ? axis : `${axis}${number}`
}

const displayTimestamp = (
  value: string,
  barInterval: ResearchReportView['bar_interval'],
): string => {
  if (barInterval === '1d') return value.slice(0, 10)
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(new Date(value))
  const part = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((item) => item.type === type)?.value ?? ''
  return `${part('year')}-${part('month')}-${part('day')} ${part('hour')}:${part('minute')}:${part('second')}`
}

const money = (value: number): string => {
  return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}
