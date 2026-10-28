import type { ReportCandidate, ResearchReportView, SegmentName, StrategyName } from '@/api/types'

const STRATEGY_ORDER: StrategyName[] = [
  'moving_average',
  'rsi',
  'macd',
  'intraday_rsi',
  'rsi_price_exit',
  'rsi_vwap_exit',
]

// Distinct hues for up to eight candidates; the selected one always takes the first.
const CANDIDATE_COLORS = {
  light: ['#2563eb', '#7c3aed', '#0d9488', '#d97706', '#db2777', '#475569', '#65a30d', '#0891b2'],
  dark: ['#60a5fa', '#a78bfa', '#2dd4bf', '#fbbf24', '#f472b6', '#94a3b8', '#a3e635', '#22d3ee'],
}

export const hasSegment = (candidate: ReportCandidate, segment: SegmentName): boolean =>
  candidate.segments.some((item) => item.segment === segment)

export const candidateSegment = (candidate: ReportCandidate, segment: SegmentName) =>
  candidate.segments.find((item) => item.segment === segment)

/** The selected candidate first, then the rest by strategy family. */
export const orderedCandidates = (report: ResearchReportView): ReportCandidate[] =>
  [...report.candidates].sort((left, right) => {
    if (left.id === report.selected_candidate_id) return -1
    if (right.id === report.selected_candidate_id) return 1
    return STRATEGY_ORDER.indexOf(left.strategy_name) - STRATEGY_ORDER.indexOf(right.strategy_name)
  })

export const defaultCandidateId = (report: ResearchReportView): string | null =>
  report.selected_candidate_id ??
  orderedCandidates(report).find((candidate) => hasSegment(candidate, 'test'))?.id ??
  report.candidates[0]?.id ??
  null

export const candidateColors = (report: ResearchReportView, dark: boolean) => {
  const palette = CANDIDATE_COLORS[dark ? 'dark' : 'light']
  return new Map(
    orderedCandidates(report).map((candidate, index) => [
      candidate.id,
      palette[index % palette.length],
    ]),
  )
}
