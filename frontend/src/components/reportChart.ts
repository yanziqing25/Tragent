import type { Data } from 'plotly.js'

type ReportTrace = Data & {
  legendgroup?: string
  meta?: { reference_role?: string; trade_detail_key?: string }
  visible?: boolean | 'legendonly'
}

export const tradeDetailKey = (trace: Data | undefined): string | null => {
  const key = (trace as ReportTrace | undefined)?.meta?.trade_detail_key
  return typeof key === 'string' ? key : null
}

export const eligibleTradeGroup = (traces: Data[]): string | null => {
  const reportTraces = traces as ReportTrace[]
  const visibleGroups = new Set(
    reportTraces
      .filter((trace) => tradeDetailKey(trace) && traceIsVisible(trace))
      .map((trace) => trace.legendgroup)
      .filter((group): group is string => typeof group === 'string'),
  )
  const assetVisible = reportTraces.some(
    (trace) => trace.meta?.reference_role === 'asset_buy_and_hold' && traceIsVisible(trace),
  )
  return assetVisible && visibleGroups.size === 1 ? [...visibleGroups][0]! : null
}

const traceIsVisible = (trace: ReportTrace): boolean => {
  return trace.visible !== false && trace.visible !== 'legendonly'
}
