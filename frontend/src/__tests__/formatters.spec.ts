import { describe, expect, it } from 'vitest'

import {
  candidateLabel,
  displayTimestamp,
  formatMoney,
  formatPercent,
  minuteTimestamp,
  studyTitle,
} from '@/lib/formatters'

describe('formatters', () => {
  it.each([
    [
      {
        strategy_name: 'rsi',
        parameters: { rsi_window: 14, entry_threshold: 40, exit_threshold: 60 },
      },
      'RSI 14 · 40/60',
    ],
    [
      { strategy_name: 'moving_average', parameters: { short_window: 5, long_window: 20 } },
      'MA 5/20',
    ],
    [
      { strategy_name: 'macd', parameters: { fast_period: 12, slow_period: 26, signal_period: 9 } },
      'MACD 12/26/9',
    ],
    [{ strategy_name: 'rsi_vwap_exit', parameters: {} }, 'RSI · VWAP entry and exit'],
  ] as const)('labels %j as %s', (candidate, label) => {
    expect(candidateLabel(candidate)).toBe(label)
  })

  it('formats money, percentages and the study title', () => {
    expect(formatMoney(1234.5, 'USD')).toBe('$1,234.50')
    expect(formatMoney(12.3, 'USD', true)).toBe('+$12.30')
    expect(formatMoney(null, 'USD')).toBe('—')
    expect(formatPercent(0.1032, { signed: true })).toBe('+10.32%')
    expect(formatPercent(-0.0269)).toBe('-2.69%')
    expect(formatPercent(null)).toBe('—')
    expect(studyTitle('AAPL', '1d', ['rsi'])).toBe('AAPL · daily RSI')
    expect(studyTitle('ZS', '1m', ['intraday_rsi', 'rsi_vwap_exit'])).toBe(
      'ZS · 1-minute intraday RSI',
    )
    expect(studyTitle(null, null, null)).toBe('Research task')
  })

  it('renders minute timestamps in US Eastern time', () => {
    expect(displayTimestamp('2024-01-02T15:30:45Z', '1m')).toBe('2024-01-02 10:30:45')
    expect(minuteTimestamp('2024-01-02T15:30:45Z')).toBe('2024-01-02 10:30')
    expect(displayTimestamp('2024-01-02T00:00:00', '1d')).toBe('2024-01-02')
  })
})
