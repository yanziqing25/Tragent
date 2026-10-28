import type { ReportCandidate, ResearchReportView } from '@/api/types'

const eastern = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hourCycle: 'h23',
})
const shortDateTime = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})

/** An amount in a currency, or a dash before it is known. Research reports are in USD. */
export const formatMoney = (
  value: number | null | undefined,
  currency: string,
  signed = false,
): string =>
  value == null
    ? '—'
    : new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency,
        signDisplay: signed ? 'exceptZero' : 'auto',
      }).format(value)

export const formatPercent = (
  value: number | null | undefined,
  { signed = false, digits = 2 } = {},
): string =>
  value == null
    ? '—'
    : new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
        signDisplay: signed ? 'exceptZero' : 'auto',
      }).format(value)

export const formatNumber = (value: number | null | undefined, digits = 2): string =>
  value == null ? '—' : value.toFixed(digits)

export const formatUpdated = (value: string): string => shortDateTime.format(new Date(value))

const easternTimestamp = (value: string): string =>
  eastern.format(new Date(value)).replace(', ', ' ')

/** Daily bars show the date; minute bars show US Eastern exchange time. */
export const displayTimestamp = (
  value: string,
  barInterval: ResearchReportView['bar_interval'],
): string => (barInterval === '1d' ? value.slice(0, 10) : easternTimestamp(value))

export const minuteTimestamp = (value: string): string => easternTimestamp(value).slice(0, -3)

const FIXED_RULES: Partial<Record<ReportCandidate['strategy_name'], string>> = {
  intraday_rsi: 'Intraday RSI baseline',
  rsi_price_exit: 'RSI · price-matched exit',
  rsi_vwap_exit: 'RSI · VWAP entry and exit',
}

export const candidateLabel = (
  candidate: Pick<ReportCandidate, 'strategy_name' | 'parameters'>,
) => {
  const fixed = FIXED_RULES[candidate.strategy_name]
  if (fixed) return fixed
  const p = candidate.parameters
  if (candidate.strategy_name === 'moving_average') return `MA ${p.short_window}/${p.long_window}`
  if (candidate.strategy_name === 'rsi')
    return `RSI ${p.rsi_window} · ${p.entry_threshold}/${p.exit_threshold}`
  return `MACD ${p.fast_period}/${p.slow_period}/${p.signal_period}`
}

const STRATEGY_FAMILY: Record<ReportCandidate['strategy_name'], string> = {
  moving_average: 'moving average',
  rsi: 'RSI',
  macd: 'MACD',
  intraday_rsi: 'intraday RSI',
  rsi_price_exit: 'intraday RSI',
  rsi_vwap_exit: 'intraday RSI',
}

/** "AAPL · daily RSI" from a goal's symbol, interval and strategies. */
export const studyTitle = (
  symbol: string | null,
  interval: '1d' | '1m' | null,
  strategies: ReportCandidate['strategy_name'][] | null,
): string => {
  if (!symbol) return 'Research task'
  const bars = interval === '1m' ? '1-minute' : interval === '1d' ? 'daily' : ''
  const families = [...new Set((strategies ?? []).map((name) => STRATEGY_FAMILY[name]))]
  const scope = families.length === 1 ? families[0] : families.length ? 'strategies' : ''
  return [symbol, [bars, scope].filter(Boolean).join(' ')].filter(Boolean).join(' · ')
}
