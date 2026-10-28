import type {
  PerformanceMetrics,
  ReportCandidate,
  ReportCandidateSegment,
  ResearchReportView,
  ResearchTaskView,
  SegmentName,
} from '@/api/types'

const DATES: Record<SegmentName, string[]> = {
  training: ['2024-01-02T00:00:00', '2024-01-03T00:00:00', '2024-01-04T00:00:00'],
  test: ['2024-01-05T00:00:00', '2024-01-08T00:00:00', '2024-01-09T00:00:00'],
}

const metrics = (total_return: number): PerformanceMetrics => ({
  total_return,
  annualized_return: null,
  annualized_volatility: null,
  sharpe_ratio: 1.5,
  max_drawdown: -0.02,
  trade_count: 1,
  turnover: 0.2,
  asset_buy_and_hold_return: 0.04,
  benchmark_return: 0.01,
})

const segment = (name: SegmentName, total: number, trade: boolean): ReportCandidateSegment => ({
  segment: name,
  metrics: metrics(total),
  equity: [1, 1.01, 1 + total],
  drawdown: [0, -0.01, 0],
  mean_exposure: 0.5,
  trades: trade
    ? [
        {
          date: DATES[name][1],
          action: 'buy',
          price: 100,
          quantity: 1,
          gross_value: 100,
          trading_cost: 0.1,
          net_cash: -100.1,
          cash_balance_after: 9899.9,
          position_quantity_after: 1,
          average_cost_after: 100.1,
          realized_profit: null,
        },
      ]
    : [],
})

const candidate = (id: string, window: number): ReportCandidate => ({
  id,
  strategy_name: 'rsi',
  parameters: { rsi_window: window, entry_threshold: 30, exit_threshold: 70 },
  transaction_cost_bps: 5,
  slippage_bps: 5,
  segments: [segment('training', 0.02, true), segment('test', 0.03, true)],
})

export const report = (): ResearchReportView => ({
  research_objective: 'Compare RSI strategies.',
  symbol: 'AAPL',
  benchmark_symbol: 'SPY',
  bar_interval: '1d',
  data_start: DATES.training[0],
  data_end: DATES.test[2],
  data_source: 'yfinance',
  selected_candidate_id: 'selected',
  stop_reason: '',
  research_incomplete: false,
  references: (['training', 'test'] as const).map((name) => ({
    segment: name,
    timestamps: DATES[name],
    asset_equity: [1, 1.02, 1.04],
    asset_drawdown: [0, 0, 0],
    benchmark_equity: [1, 1.005, 1.01],
    benchmark_drawdown: [0, 0, 0],
  })),
  candidates: [candidate('alternative', 7), candidate('selected', 14)],
  initial_capital: 10_000,
  warnings: [],
  limitations: [],
  critic_reviews: [],
  critic_review_status: 'accepted',
  market_context_summary: null,
  market_context_sources: [],
})

export const taskView = (overrides: Partial<ResearchTaskView> = {}): ResearchTaskView => ({
  thread_id: '11111111-1111-4111-8111-111111111111',
  status: 'running',
  run_state: 'running',
  can_resume: false,
  goal: null,
  missing_goal_fields: [],
  clarification_hints: [],
  clarification_example: null,
  defaulted_fields: [],
  budget_used: { agent_steps: 1, tool_calls: 1, training_backtests: 0 },
  budget_remaining: { agent_steps: 39, tool_calls: 29, training_backtests: 11 },
  active_agent: 'research',
  critic_round: 0,
  critic_verdict: null,
  critic_budget_used: { agent_steps: 0, tool_calls: 0 },
  critic_budget_remaining: { agent_steps: 10, tool_calls: 6 },
  last_agent: 'research',
  last_action: 'get_market_data',
  last_observation: null,
  selected_fingerprint: null,
  stop_reason: null,
  report_exists: false,
  safe_error: null,
  ...overrides,
})
