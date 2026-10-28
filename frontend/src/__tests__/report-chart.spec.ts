import { describe, expect, it } from 'vitest'
import type { LineSeriesOption, TooltipComponentFormatterCallbackParams } from 'echarts'

import { buildSegmentOption, CHART_PALETTE, type ChartState } from '@/lib/reportChart'
import { candidateColors } from '@/lib/report'
import { report } from './fixtures'

const state = (overrides: Partial<ChartState> = {}): ChartState => ({
  visible: ['selected'],
  showAsset: true,
  showBenchmark: true,
  metric: 'equity',
  ...overrides,
})

const build = (overrides: Partial<ChartState> = {}) => {
  const value = report()
  const option = buildSegmentOption(
    value,
    'test',
    state(overrides),
    candidateColors(value, false),
    CHART_PALETTE.light,
  )!
  const series = option.series as LineSeriesOption[]
  return { option, series, names: series.map((item) => item.name) }
}

const hover = (overrides: Partial<ChartState> = {}) => {
  const { option, series } = build(overrides)
  const params = series.map((item, index) => ({
    seriesName: item.name,
    name: '2024-01-08',
    color: '#000',
    data: (item.data as unknown[])[1],
    seriesIndex: index,
  }))
  const formatter = (option.tooltip as { formatter: (raw: unknown) => string }).formatter
  return formatter(params as unknown as TooltipComponentFormatterCallbackParams)
}

describe('report segment chart', () => {
  it('draws only the visible strategies and references', () => {
    expect(build().names).toEqual(['AAPL buy and hold', 'SPY', 'RSI 14 · 30/70'])
    expect(build({ visible: ['selected', 'alternative'], showBenchmark: false }).names).toEqual([
      'AAPL buy and hold',
      'RSI 14 · 30/70',
      'RSI 7 · 30/70',
    ])
  })

  it.each([
    [{}, true],
    [{ showBenchmark: false }, true],
    [{ visible: ['selected', 'alternative'] }, false],
    [{ showAsset: false }, false],
    [{ metric: 'drawdown' as const }, false],
  ])(
    'tags trades on the asset line only for one strategy with the asset shown: %j',
    (overrides, tagged) => {
      const { series } = build(overrides)
      const asset = series.find((item) => item.name === 'AAPL buy and hold')
      const marks = asset?.markPoint?.data ?? []
      expect(marks.length > 0).toBe(tagged)
      if (tagged) {
        expect(marks[0]).toMatchObject({ value: 'B', coord: ['2024-01-08', 1.02] })
        expect(hover(overrides)).toContain('Buy · RSI 14 · 30/70')
      } else {
        expect(hover(overrides)).not.toContain('Buy ·')
      }
    },
  )

  it('switches every line to drawdown values and formats them as percentages', () => {
    const { series } = build({ metric: 'drawdown' })
    const strategy = series.find((item) => item.name === 'RSI 14 · 30/70')!
    expect((strategy.data as { value: number }[]).map((point) => point.value)).toEqual([
      0, -0.01, 0,
    ])
    expect(hover({ metric: 'drawdown' })).toContain('-1.00%')
  })
})
