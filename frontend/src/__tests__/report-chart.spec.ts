import { describe, expect, it } from 'vitest'
import type { Data } from 'plotly.js'

import type { ResearchReportView } from '@/api/generated'
import { eligibleTradeGroup, tradeDetailKey } from '@/components/reportChart'
import { buildReportFigure } from '@/components/reportFigure'

describe('report chart interaction rules', () => {
  it('shows trade overlays only for asset Buy and Hold plus exactly one strategy', () => {
    const asset = trace({ meta: { reference_role: 'asset_buy_and_hold' } })
    const selected = trace({
      legendgroup: 'selected',
      meta: { trade_detail_key: 'selected' },
    })
    const alternative = trace({
      legendgroup: 'alternative',
      meta: { trade_detail_key: 'alternative' },
      visible: 'legendonly',
    })

    expect(eligibleTradeGroup([asset, selected, alternative])).toBe('selected')
    expect(tradeDetailKey(selected)).toBe('selected')

    alternative.visible = true
    expect(eligibleTradeGroup([asset, selected, alternative])).toBeNull()

    asset.visible = 'legendonly'
    alternative.visible = 'legendonly'
    expect(eligibleTradeGroup([asset, selected, alternative])).toBeNull()
  })

  it('builds Plotly presentation from semantic report facts', () => {
    const figure = buildReportFigure(report())
    const traces = figure.data as ChartTrace[]
    const selected = traces.filter((item) => item.legendgroup === 'selected')
    const alternative = traces.filter((item) => item.legendgroup === 'alternative')

    expect(selected).toHaveLength(4)
    expect(selected.every((item) => item.visible === true)).toBe(true)
    expect(alternative.every((item) => item.visible === 'legendonly')).toBe(true)
    expect(figure.layout.shapes?.some((shape) =>
      shape.legendgroup === 'selected' && shape.visible === true)).toBe(true)
    expect(figure.config.scrollZoom).toBe(true)
  })
})

const trace = (
  value: Record<string, unknown>,
): Data & { visible?: boolean | 'legendonly' } => {
  return { type: 'scatter', ...value } as Data & { visible?: boolean | 'legendonly' }
}

type ChartTrace = Data & {
  legendgroup?: string
  visible?: boolean | 'legendonly'
}

const report = (): ResearchReportView => ({
  research_objective: 'Compare strategies.',
  symbol: 'AAPL',
  benchmark_symbol: 'SPY',
  bar_interval: '1d',
  data_start: '2024-01-01T00:00:00',
  data_end: '2024-01-02T00:00:00',
  data_source: 'yfinance',
  selected_candidate_id: 'selected',
  stop_reason: '',
  research_incomplete: false,
  references: ['training', 'test'].map((segment) => ({
    segment: segment as 'training' | 'test',
    timestamps: ['2024-01-01T00:00:00', '2024-01-02T00:00:00'],
    asset_equity: [1, 1.02],
    asset_drawdown: [0, 0],
    benchmark_equity: [1, 1.01],
    benchmark_drawdown: [0, 0],
  })),
  candidates: ['selected', 'alternative'].map((id, index) => ({
    id,
    strategy_name: 'moving_average',
    parameters: { short_window: 2 + index, long_window: 5 + index },
    transaction_cost_bps: 5,
    slippage_bps: 5,
    segments: ['training', 'test'].map((segment) => ({
      segment: segment as 'training' | 'test',
      equity: [1, 1.01 + index / 100],
      drawdown: [0, -0.01],
      metrics: {
        total_return: 0.01,
        annualized_return: 0.1,
        annualized_volatility: 0.2,
        sharpe_ratio: 0.5,
        max_drawdown: -0.01,
        trade_count: 1,
        turnover: 1,
        asset_buy_and_hold_return: 0.02,
        benchmark_return: 0.01,
      },
      trades: [{
        date: '2024-01-02T00:00:00',
        action: 'buy',
        price: 100,
        quantity: 1,
        gross_value: 100,
        trading_cost: 0.1,
        net_cash: -100.1,
        realized_profit: null,
        average_cost_after: 100.1,
        position_quantity_after: 1,
        cash_balance_after: 9_899.9,
      }],
    })),
  })),
  initial_capital: 10_000,
  warnings: [],
  limitations: [],
  critic_reviews: [],
  critic_review_status: 'accepted',
})
