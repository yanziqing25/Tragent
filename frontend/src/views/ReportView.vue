<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'

import { getResearchReport } from '@/api/api'
import ReportChart from '@/components/ReportChart.vue'
import TradeDetails from '@/components/TradeDetails.vue'
import type { PerformanceMetrics, ReportCandidate, ResearchReportView } from '@/api/generated'
import { candidateLabel, defaultReportCandidateId } from '@/components/reportFigure'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const route = useRoute()
const threadId = String(route.params.threadId)
const report = ref<ResearchReportView | null>(null)
const error = ref<string | null>(null)
const loading = ref(true)
const tradeDetails = ref<InstanceType<typeof TradeDetails> | null>(null)
const controller = new AbortController()

const metricColumns = [
  ['total_return', 'Total return'],
  ['annualized_return', 'Annualized return'],
  ['annualized_volatility', 'Annualized volatility'],
  ['sharpe_ratio', 'Sharpe ratio'],
  ['max_drawdown', 'Max drawdown'],
  ['trade_count', 'Trade count'],
  ['turnover', 'Turnover'],
  ['asset_buy_and_hold_return', 'Asset buy-and-hold return'],
  ['benchmark_return', 'Benchmark buy-and-hold return'],
] as const satisfies ReadonlyArray<readonly [keyof PerformanceMetrics, string]>

type MetricRow = {
  candidate: ReportCandidate
  segment: 'training' | 'test'
  metrics: PerformanceMetrics | null
}

const selectedCandidate = computed(() =>
  report.value?.candidates.find((candidate) => candidate.id === report.value?.selected_candidate_id),
)
const defaultCandidateId = computed(() =>
  report.value ? defaultReportCandidateId(report.value) : null,
)
const hasFinalEvaluation = computed(() =>
  report.value?.candidates.some((candidate) =>
    candidate.segments.some((segment) => segment.segment === 'test'),
  ) ?? false,
)
const metricRows = computed<MetricRow[]>(() =>
  (report.value?.candidates ?? []).flatMap((candidate) => {
    const rows: MetricRow[] = candidate.segments.map((segment) => ({
      candidate,
      segment: segment.segment,
      metrics: segment.metrics,
    }))
    if (!candidate.segments.some((segment) => segment.segment === 'test')) {
      rows.push({ candidate, segment: 'test', metrics: null })
    }
    return rows
  }),
)
const benchmarkLabel = computed(() => {
  if (report.value?.bar_interval === '1m') return 'Session Buy and Hold'
  return report.value?.benchmark_symbol ?? 'No benchmark'
})
const dataRange = computed(() => {
  if (!report.value?.data_start || !report.value.data_end) return 'N/A'
  if (report.value.bar_interval === '1d') {
    return `${report.value.data_start.slice(0, 10)} to ${report.value.data_end.slice(0, 10)}`
  }
  return `${minuteTimestamp(report.value.data_start)} to ${minuteTimestamp(report.value.data_end)} US/Eastern`
})
const dataSourceLabel = computed(() =>
  report.value?.bar_interval === '1m'
    ? 'yfinance adjusted 1-minute regular-session bars'
    : 'yfinance adjusted daily bars',
)
const criticFinalVerdict = computed(() => {
  if (report.value?.critic_review_status === 'accepted') return 'ACCEPT'
  if (report.value?.critic_review_status === 'rejected') return 'REJECT'
  return 'N/A — Critic review did not finish.'
})

onMounted(async () => {
  try {
    report.value = await getResearchReport(threadId, controller.signal)
  } catch (cause) {
    if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
      if (!auth.handleUnauthorized(cause)) {
        error.value = cause instanceof Error ? cause.message : 'Report loading failed.'
      }
    }
  } finally {
    loading.value = false
  }
})

onUnmounted(() => controller.abort())

const focusTradeDetails = (candidateKey: string) => {
  tradeDetails.value?.focusCandidate(candidateKey)
}

const parameters = (value: Record<string, number>): string => {
  return Object.entries(value)
    .map(([name, parameter]) => `${name}=${parameter}`)
    .join(', ')
}

const costs = (transactionCost: number, slippage: number): string => {
  return `${transactionCost} + ${slippage} bps`
}

const metric = (name: string, value: number | null | undefined): string => {
  if (value == null) return 'N/A'
  if (name === 'trade_count') return String(Math.trunc(value))
  if (
    [
      'total_return',
      'annualized_return',
      'annualized_volatility',
      'max_drawdown',
      'asset_buy_and_hold_return',
      'benchmark_return',
    ].includes(name)
  ) {
    return `${(value * 100).toFixed(2)}%`
  }
  return value.toFixed(4)
}

const segmentLabel = (segment: 'training' | 'test'): string => {
  return segment === 'training' ? 'In-sample backtest' : 'Out-of-sample validation'
}

const minuteTimestamp = (value: string): string => {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  }).format(new Date(value)).replace(',', '')
}

</script>

<template>
  <v-container class="page-shell report-page" max-width="1440">
    <v-btn
      class="mb-7"
      variant="text"
      :to="{ name: 'research-detail', params: { threadId } }"
    >
      ← Back to task
    </v-btn>

    <v-progress-linear v-if="loading" color="primary" indeterminate rounded />
    <v-alert v-else-if="error" type="error" variant="tonal">{{ error }}</v-alert>

    <template v-else-if="report">
      <v-card class="report-hero mb-7">
        <v-card-text class="pa-6 pa-sm-8">
          <div class="page-eyebrow mb-3">Agentic US Equity Research</div>
          <h1 class="report-title mb-3">{{ report.symbol }} Research Report</h1>
          <p class="page-copy mb-5">{{ report.research_objective }}</p>
          <v-chip
            :color="selectedCandidate ? 'success' : 'warning'"
            size="large"
            variant="tonal"
          >
            {{
              selectedCandidate
                ? `Selected · ${candidateLabel(selectedCandidate)}`
                : `No strategy selected · ${report.stop_reason}`
            }}
          </v-chip>

          <v-row class="mt-5">
            <v-col cols="12" sm="6" lg="3">
              <v-sheet class="summary-tile pa-4" rounded="lg">
                <div class="detail-label">Asset</div>
                <strong>{{ report.symbol }}</strong>
              </v-sheet>
            </v-col>
            <v-col cols="12" sm="6" lg="3">
              <v-sheet class="summary-tile pa-4" rounded="lg">
                <div class="detail-label">Benchmark</div>
                <strong>{{ benchmarkLabel }}</strong>
              </v-sheet>
            </v-col>
            <v-col cols="12" sm="6" lg="3">
              <v-sheet class="summary-tile pa-4" rounded="lg">
                <div class="detail-label">Data range</div>
                <strong>{{ dataRange }}</strong>
              </v-sheet>
            </v-col>
            <v-col cols="12" sm="6" lg="3">
              <v-sheet class="summary-tile pa-4" rounded="lg">
                <div class="detail-label">Data source</div>
                <strong>{{ dataSourceLabel }}</strong>
              </v-sheet>
            </v-col>
          </v-row>

          <v-alert v-if="report.research_incomplete" class="mt-5" type="warning" variant="tonal">
            <strong>Research incomplete:</strong> Out-of-sample validation was not completed.
            In-sample results are shown for audit only.
          </v-alert>
        </v-card-text>
      </v-card>

      <v-card class="surface-card mb-7">
        <v-card-text class="pa-6">
          <div class="section-kicker mb-2">Interactive evidence</div>
          <h2 class="section-title mb-2">Equity and Drawdown curves</h2>
          <p class="text-medium-emphasis mb-5">
            Asset Buy and Hold and one default strategy are visible initially. B/S transaction
            tags and trade-date hover appear only while the asset reference and exactly one
            strategy are visible. Showing multiple strategies or hiding the asset reference hides
            both transaction displays.
          </p>
          <ReportChart
            v-if="hasFinalEvaluation"
            :report="report"
            @focus-trade-details="focusTradeDetails"
          />
          <v-empty-state v-else title="No final evaluation results" />
        </v-card-text>
      </v-card>

      <v-card class="surface-card mb-7">
        <v-card-text class="pa-6 pb-3">
          <div class="section-kicker mb-2">Deterministic comparison</div>
          <h2 class="section-title mb-2">Strategy metrics</h2>
          <p class="text-medium-emphasis">
            All evaluated candidates remain visible; selected rows are highlighted.
          </p>
        </v-card-text>
        <div class="report-table-shell">
          <v-table class="report-table" density="compact">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Segment</th>
                <th>Parameters</th>
                <th>Costs</th>
                <th v-for="([name, label]) in metricColumns" :key="name">{{ label }}</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in metricRows"
                :key="`${row.candidate.id}-${row.segment}`"
                :class="{ 'selected-row': row.candidate.id === report.selected_candidate_id }"
              >
                <td>{{ candidateLabel(row.candidate) }}</td>
                <td>{{ segmentLabel(row.segment) }}</td>
                <td>{{ parameters(row.candidate.parameters) }}</td>
                <td>
                  {{ costs(row.candidate.transaction_cost_bps, row.candidate.slippage_bps) }}
                </td>
                <td v-for="([name]) in metricColumns" :key="name">
                  {{ metric(name, row.metrics?.[name]) }}
                </td>
              </tr>
              <tr v-if="metricRows.length === 0">
                <td :colspan="4 + metricColumns.length">No completed backtests.</td>
              </tr>
            </tbody>
          </v-table>
        </div>
      </v-card>

      <TradeDetails
        ref="tradeDetails"
        :candidates="report.candidates"
        :default-candidate-id="defaultCandidateId"
        :selected-candidate-id="report.selected_candidate_id"
        :initial-capital="report.initial_capital"
        :bar-interval="report.bar_interval"
      />

      <v-card v-if="report.critic_review_status" class="surface-card mb-7">
        <v-card-text class="pa-6">
          <div class="section-kicker mb-2">Independent Agent review</div>
          <h2 class="section-title mb-2">AI Critic Review</h2>
          <p class="text-medium-emphasis mb-5">
            The Critic Agent reviewed the frozen Research draft without recomputing the backtest.
          </p>
          <v-chip color="secondary" variant="tonal">
            Final verdict · {{ criticFinalVerdict }}
          </v-chip>
          <v-sheet
            v-for="(review, index) in report.critic_reviews"
            :key="index"
            class="summary-tile pa-5 mt-4"
            rounded="lg"
          >
            <div class="detail-label mb-2">Round {{ index + 1 }} · {{ review.verdict }}</div>
            <strong>{{ review.summary }}</strong>
            <ul v-if="review.issues?.length" class="mt-3">
              <li v-for="issue in review.issues" :key="issue">{{ issue }}</li>
            </ul>
            <p v-if="review.required_revision" class="mt-3 mb-0">
              <strong>Requested revision:</strong> {{ review.required_revision }}
            </p>
          </v-sheet>
          <v-alert v-if="report.critic_reviews.length === 0" class="mt-4" type="warning" variant="tonal">
            Critic validation did not finish; no verdict was fabricated.
          </v-alert>
        </v-card-text>
      </v-card>

      <v-row>
        <v-col cols="12" md="6">
          <v-card class="surface-card h-100">
            <v-card-text class="pa-6">
              <h2 class="section-title mb-4">Warnings</h2>
              <ul v-if="report.warnings.length"><li v-for="item in report.warnings" :key="item">{{ item }}</li></ul>
              <p v-else class="text-medium-emphasis">No warnings.</p>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" md="6">
          <v-card class="surface-card h-100">
            <v-card-text class="pa-6">
              <h2 class="section-title mb-4">Limitations</h2>
              <ul v-if="report.limitations.length"><li v-for="item in report.limitations" :key="item">{{ item }}</li></ul>
              <p v-else class="text-medium-emphasis">No additional limitations were supplied.</p>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <footer class="report-footer mt-7">
        <template v-if="report.bar_interval === '1m'">
          This report uses adjusted 1-minute XNYS regular-session data, next-minute-open execution,
          user-confirmed session counts, daily forced liquidation, independent sample-in/sample-out
          capital, and Session Buy and Hold. Indicator state and the logical target may continue
          across sessions, but actual positions never remain overnight. Annualized return,
          annualized volatility, and Sharpe ratio are N/A.
        </template>
        <template v-else>
          This report uses adjusted daily data, next-open execution, a 70/30 chronological split,
          a 0% risk-free rate, and deterministic calculations.
        </template>
        “In-sample” means research on the earlier time segment; no machine-learning model is
        trained. Past performance is not a guarantee of future results.
      </footer>
    </template>
  </v-container>
</template>

<style scoped>
.report-page {
  padding-bottom: 4rem;
}

.report-hero {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  background: rgb(var(--v-theme-surface));
}

.report-title {
  max-width: 60rem;
  font-size: clamp(2rem, 5vw, 3.5rem);
  line-height: 1.02;
  letter-spacing: -0.045em;
}

.summary-tile {
  height: 100%;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  background: rgb(var(--v-theme-surface-variant));
}

.report-table-shell {
  overflow-x: auto;
}

.report-table {
  min-width: 79rem;
}

.report-table th,
.report-table td {
  white-space: nowrap;
}

.report-table th:first-child,
.report-table td:first-child,
.report-table th:nth-child(2),
.report-table td:nth-child(2),
.report-table th:nth-child(3),
.report-table td:nth-child(3) {
  text-align: left;
}

.selected-row td {
  background: rgba(var(--v-theme-primary), 0.1);
}

.report-footer {
  padding: 1.25rem;
  color: rgb(var(--v-theme-on-surface-variant));
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 1rem;
  background: rgba(var(--v-theme-surface), 0.75);
  font-size: 0.875rem;
}

li + li {
  margin-top: 0.4rem;
}
</style>
