<script setup lang="ts">
import { ref } from 'vue'

import type { ReportCandidate } from '@/api/generated'
import { candidateLabel } from '@/components/reportFigure'

const props = defineProps<{
  candidates: ReportCandidate[]
  defaultCandidateId: string | null
  selectedCandidateId: string | null
  initialCapital: number
  barInterval: '1d' | '1m'
}>()

const openCandidateIds = ref<string[]>(props.defaultCandidateId ? [props.defaultCandidateId] : [])
const openSegments = ref<Record<string, string[]>>(
  Object.fromEntries(
    props.candidates.map((candidate) => {
      const segment = candidate.segments.find((item) => item.segment === 'test')
        ?? candidate.segments[0]
      return [candidate.id, segment ? [segment.segment] : []]
    }),
  ),
)

const focusCandidate = (candidateId: string) => {
  openCandidateIds.value = [candidateId]
  if (openSegments.value[candidateId]?.length) return
  const candidate = props.candidates.find((item) => item.id === candidateId)
  const segment = candidate?.segments.find((item) => item.segment === 'test')
    ?? candidate?.segments[0]
  openSegments.value[candidateId] = segment ? [segment.segment] : []
}

const setOpenCandidates = (value: unknown) => {
  openCandidateIds.value = stringArray(value)
}

const setOpenSegments = (candidateId: string, value: unknown) => {
  openSegments.value[candidateId] = stringArray(value)
}

const stringArray = (value: unknown): string[] => {
  return Array.isArray(value) ? value.map(String) : []
}

const tradeCount = (candidate: ReportCandidate): number => {
  return candidate.segments.reduce((total, segment) => total + segment.trades.length, 0)
}

const money = (value: number, signed = false): string => {
  const absolute = Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
  const sign = signed ? (value > 0 ? '+' : value < 0 ? '-' : '') : ''
  return `${sign}$${absolute}`
}

const tone = (value: number | null): string => {
  if (value === null || value === 0) return 'profit-neutral'
  return value > 0 ? 'profit-positive' : 'profit-negative'
}

const segmentLabel = (segment: 'training' | 'test'): string => {
  return segment === 'training' ? 'In-sample backtest' : 'Out-of-sample validation'
}

const tradeDate = (value: string): string => {
  if (props.barInterval === '1d') return value.slice(0, 10)
  return `${new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  }).format(new Date(value)).replace(',', '')} ET`
}

defineExpose({ focusCandidate })
</script>

<template>
  <v-card class="surface-card trade-ledger mb-7">
    <v-card-text class="pa-6 pb-3">
      <div class="section-kicker mb-2">Execution ledger</div>
      <h2 class="section-title mb-2">Trade details</h2>
      <p class="text-medium-emphasis">
        Each Buy or Sell is one transaction. Every segment starts with an independently funded
        {{ money(initialCapital) }} illustration and uses fractional shares. Prices are raw
        next-open execution prices; costs include fees and slippage.
      </p>
    </v-card-text>

    <v-expansion-panels
      v-if="candidates.length"
      class="trade-candidates px-4 px-sm-6 pb-6"
      multiple
      :model-value="openCandidateIds"
      @update:model-value="setOpenCandidates"
    >
      <v-expansion-panel
        v-for="candidate in candidates"
        :key="candidate.id"
        class="trade-candidate"
        :class="{ 'trade-candidate--selected': candidate.id === selectedCandidateId }"
        :value="candidate.id"
      >
        <v-expansion-panel-title class="trade-candidate-title">
          <div class="d-flex flex-wrap align-center justify-space-between ga-4 w-100 pr-3">
            <div>
              <div class="d-flex flex-wrap align-center ga-2 mb-1">
                <strong class="trade-candidate-name">{{ candidateLabel(candidate) }}</strong>
                <v-chip
                  v-if="candidate.id === selectedCandidateId"
                  color="primary"
                  size="small"
                  variant="flat"
                >
                  Selected strategy
                </v-chip>
              </div>
              <span class="text-caption text-medium-emphasis">
                Training and test execution ledger
              </span>
            </div>
            <div class="trade-count text-right">
              <strong>{{ tradeCount(candidate) }}</strong>
              <span>transactions</span>
            </div>
          </div>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <v-expansion-panels
            class="trade-segments"
            multiple
            variant="accordion"
            :model-value="openSegments[candidate.id] ?? []"
            @update:model-value="setOpenSegments(candidate.id, $event)"
          >
            <v-expansion-panel
              v-for="segment in candidate.segments"
              :key="segment.segment"
              class="trade-segment"
              :value="segment.segment"
            >
              <v-expansion-panel-title class="trade-segment-title">
                <div class="d-flex flex-wrap align-center justify-space-between ga-3 w-100 pr-3">
                  <div>
                    <strong>{{ segmentLabel(segment.segment) }}</strong>
                    <div class="text-caption text-medium-emphasis">
                      {{
                        segment.segment === 'test'
                          ? 'Out-of-sample execution'
                          : 'In-sample execution'
                      }}
                    </div>
                  </div>
                  <v-chip size="small" variant="tonal">
                    {{ segment.trades.length }} trades
                  </v-chip>
                </div>
              </v-expansion-panel-title>
              <v-expansion-panel-text>
                <div v-if="segment.trades.length" class="report-table-shell trade-table-shell">
                  <v-table class="trade-table" density="compact">
                    <thead>
                      <tr>
                        <th>#</th><th>Date</th><th>Action</th><th>Price</th>
                        <th>Quantity</th><th>Gross value</th><th>Trading cost</th>
                        <th>Net cash</th><th>Realized P/L</th><th>Average cost</th>
                        <th>Position quantity</th><th>Cash balance</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(trade, index) in segment.trades" :key="index">
                        <td class="trade-index">{{ index + 1 }}</td>
                        <td class="trade-date">{{ tradeDate(trade.date) }}</td>
                        <td>
                          <v-chip
                            :color="trade.action === 'buy' ? 'success' : 'error'"
                            size="x-small"
                            variant="tonal"
                          >
                            {{ trade.action.toUpperCase() }}
                          </v-chip>
                        </td>
                        <td class="trade-number">{{ money(trade.price) }}</td>
                        <td class="trade-number">{{ trade.quantity.toFixed(4) }}</td>
                        <td class="trade-number">{{ money(trade.gross_value) }}</td>
                        <td class="trade-number">{{ money(trade.trading_cost) }}</td>
                        <td class="trade-number" :class="tone(trade.net_cash)">
                          {{ money(trade.net_cash, true) }}
                        </td>
                        <td class="trade-number" :class="tone(trade.realized_profit)">
                          {{
                            trade.realized_profit === null
                              ? 'N/A'
                              : money(trade.realized_profit, true)
                          }}
                        </td>
                        <td class="trade-number">
                          {{
                            trade.average_cost_after === null
                              ? 'N/A'
                              : money(trade.average_cost_after)
                          }}
                        </td>
                        <td class="trade-number">
                          {{ trade.position_quantity_after.toFixed(4) }}
                        </td>
                        <td class="trade-number trade-balance">
                          {{ money(trade.cash_balance_after) }}
                        </td>
                      </tr>
                    </tbody>
                  </v-table>
                </div>
                <p v-else class="text-medium-emphasis pa-2">
                  No executed trades in this segment.
                </p>
              </v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
    <v-card-text v-else class="text-medium-emphasis">No executed trades.</v-card-text>
  </v-card>
</template>

<style scoped>
.trade-ledger {
  overflow: hidden;
}

.trade-candidates {
  gap: 0.75rem;
}

.trade-candidate {
  overflow: hidden;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  background: rgba(var(--v-theme-surface-variant), 0.24) !important;
}

.trade-candidate--selected {
  border-color: rgba(var(--v-theme-primary), 0.38);
  box-shadow: inset 3px 0 rgb(var(--v-theme-primary)) !important;
}

.trade-candidate-title {
  min-height: 5.25rem;
}

.trade-candidate-name {
  font-size: 1rem;
  letter-spacing: -0.01em;
}

.trade-count strong,
.trade-count span {
  display: block;
}

.trade-count strong {
  color: rgb(var(--v-theme-on-surface));
  font-size: 1.1rem;
  font-variant-numeric: tabular-nums;
}

.trade-count span {
  color: rgb(var(--v-theme-on-surface-variant));
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.trade-segments {
  overflow: hidden;
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 0.9rem;
}

.trade-segment {
  background: rgb(var(--v-theme-surface)) !important;
}

.trade-segment + .trade-segment {
  border-top: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}

.trade-segment-title {
  min-height: 4rem;
}

.report-table-shell {
  overflow-x: auto;
}

.trade-table-shell {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 0.75rem;
}

.trade-table {
  min-width: 94rem;
  font-variant-numeric: tabular-nums;
}

.trade-table th,
.trade-table td {
  white-space: nowrap;
}

.trade-table th {
  height: 2.75rem !important;
  color: rgb(var(--v-theme-on-surface-variant));
  background: rgba(var(--v-theme-surface-variant), 0.72);
  font-size: 0.68rem;
  font-weight: 800 !important;
  letter-spacing: 0.045em;
  text-transform: uppercase;
}

.trade-table td {
  height: 3.25rem !important;
  border-bottom-color: rgba(var(--v-border-color), 0.6) !important;
}

.trade-table tbody tr:hover td {
  background: rgba(var(--v-theme-primary), 0.045);
}

.trade-table th:nth-child(n + 4),
.trade-table td:nth-child(n + 4),
.trade-number {
  text-align: right;
}

.trade-index {
  width: 3rem;
  color: rgb(var(--v-theme-on-surface-variant));
  text-align: center !important;
}

.trade-date {
  font-weight: 650;
}

.trade-balance {
  font-weight: 750;
}

.profit-positive {
  color: rgb(var(--v-theme-success));
  font-weight: 700;
}

.profit-negative {
  color: rgb(var(--v-theme-error));
  font-weight: 700;
}

.profit-neutral {
  color: rgb(var(--v-theme-on-surface-variant));
  font-weight: 700;
}
</style>
