<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ToggleGroupItem, ToggleGroupRoot } from 'reka-ui'
import { ArrowLeft, Check, CircleAlert, Minus } from '@lucide/vue'

import type { ReportCandidate } from '@/api/types'
import CandidateTable from '@/components/report/CandidateTable.vue'
import SegmentChart from '@/components/report/SegmentChart.vue'
import TradeLedger from '@/components/report/TradeLedger.vue'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { errorMessage } from '@/lib/errors'
import { candidateLabel, formatNumber, formatPercent, minuteTimestamp } from '@/lib/formatters'
import {
  candidateColors,
  candidateSegment,
  defaultCandidateId,
  orderedCandidates,
} from '@/lib/report'
import {
  assetName,
  buildSegmentOption,
  CHART_PALETTE,
  tagsVisible,
  type ChartMetric,
  type ChartState,
} from '@/lib/reportChart'
import { CRITIC_VERDICT, criticReviewLabel } from '@/lib/status'
import { useTheme } from '@/lib/theme'
import { useReport } from '@/queries/research'
import DeployStrategy from '@/components/trading/DeployStrategy.vue'

const SOURCE_LABELS = { official: 'Official', news: 'News', community: 'Community' } as const

const route = useRoute()
const threadId = computed(() => String(route.params.threadId))
const query = useReport(threadId)
const report = computed(() => query.data.value)
// Only a daily strategy the Critic accepted can trade the paper account.
const deployable = computed(
  () =>
    report.value?.critic_review_status === 'accepted' &&
    report.value.selected_candidate_id !== null &&
    report.value.bar_interval === '1d',
)
const { isDark } = useTheme()

const candidates = computed(() => (report.value ? orderedCandidates(report.value) : []))
const colors = computed(() =>
  report.value ? candidateColors(report.value, isDark.value) : new Map<string, string>(),
)
const palette = computed(() => CHART_PALETTE[isDark.value ? 'dark' : 'light'])

// The legend and the trade ledger share one selection.
const visible = ref<string[]>([])
const showAsset = ref(true)
const showBenchmark = ref(true)
const metric = ref<ChartMetric>('equity')
const openId = ref<string | null>(null)

watch(
  () => report.value?.selected_candidate_id,
  () => {
    if (!report.value) return
    const initial = defaultCandidateId(report.value)
    visible.value = initial ? [initial] : []
    openId.value = initial
  },
  { immediate: true },
)

const toggleCandidate = (id: string) => {
  if (!visible.value.includes(id)) {
    visible.value = [...visible.value, id]
    openId.value = id
    return
  }
  const rest = visible.value.filter((item) => item !== id)
  visible.value = rest
  if (openId.value === id) openId.value = rest[rest.length - 1] ?? null
}

/** Opening a ledger shows that strategy alone, like picking it in the legend. */
const toggleLedger = (id: string) => {
  if (openId.value === id) {
    openId.value = null
    return
  }
  openId.value = id
  visible.value = [id]
}

const state = computed<ChartState>(() => ({
  visible: visible.value,
  showAsset: showAsset.value,
  showBenchmark: showBenchmark.value,
  metric: metric.value,
}))

const primary = computed<ReportCandidate | undefined>(() =>
  candidates.value.find((candidate) => candidate.id === defaultCandidateId(report.value!)),
)
const primaryTest = computed(() => primary.value && candidateSegment(primary.value, 'test'))
const headline = computed(
  () => primaryTest.value ?? (primary.value && candidateSegment(primary.value, 'training')),
)

const stats = computed(() => {
  const segment = headline.value
  if (!segment) return []
  const m = segment.metrics
  return [
    { label: 'Return', value: formatPercent(m.total_return, { signed: true }) },
    { label: 'Sharpe', value: formatNumber(m.sharpe_ratio) },
    { label: 'Max drawdown', value: formatPercent(m.max_drawdown) },
    { label: 'Trades', value: String(m.trade_count) },
    { label: 'Exposure', value: formatPercent(segment.mean_exposure, { digits: 1 }) },
  ]
})

const comparison = computed(() => {
  const segment = primaryTest.value
  if (!segment || !report.value || !primary.value) return []
  const m = segment.metrics
  const rows = [
    {
      label: candidateLabel(primary.value),
      value: m.total_return,
      color: colors.value.get(primary.value.id)!,
    },
    {
      label: assetName(report.value),
      value: m.asset_buy_and_hold_return,
      color: palette.value.asset,
    },
  ]
  if (m.benchmark_return !== null && report.value.benchmark_symbol) {
    rows.push({
      label: report.value.benchmark_symbol,
      value: m.benchmark_return,
      color: palette.value.benchmark,
    })
  }
  const largest = Math.max(...rows.map((row) => Math.abs(row.value)), 1e-9)
  return rows.map((row, index) => ({
    ...row,
    text: formatPercent(row.value, { signed: true }),
    width: `${(Math.abs(row.value) / largest) * 100}%`,
    delay: `${200 + index * 100}ms`,
  }))
})

const verdict = computed(() => {
  const value = report.value
  if (!value) return null
  const selected = candidates.value.find(
    (candidate) => candidate.id === value.selected_candidate_id,
  )
  const critic = criticReviewLabel(value.critic_review_status)
  if (!selected) {
    return {
      tone: value.critic_review_status === 'rejected' ? 'danger' : 'neutral',
      title: `No strategy selected · ${critic}`,
      body: value.stop_reason,
    }
  }
  const m = primaryTest.value?.metrics
  const points = (a: number, b: number) => `${formatNumber(Math.abs(a - b) * 100)} points`
  const parts: string[] = []
  if (m && m.benchmark_return !== null && value.benchmark_symbol) {
    parts.push(
      `${m.total_return >= m.benchmark_return ? 'Beat' : 'Trailed'} ${value.benchmark_symbol} by ${points(m.total_return, m.benchmark_return)}`,
    )
  }
  if (m) {
    const verb = m.total_return >= m.asset_buy_and_hold_return ? 'beat' : 'trailed'
    parts.push(
      `${parts.length ? verb : verb[0].toUpperCase() + verb.slice(1)} holding ${value.symbol} by ${points(m.total_return, m.asset_buy_and_hold_return)}`,
    )
  }
  const exposure = primaryTest.value
    ? `, with ${formatPercent(primaryTest.value.mean_exposure, { digits: 0 })} average exposure`
    : ''
  return {
    tone: 'success',
    title: `Selected ${candidateLabel(selected)} · ${critic}`,
    body: parts.length ? `${parts.join(' and ')} out of sample${exposure}.` : '',
  }
})

const legend = computed(() => {
  const value = report.value
  if (!value) return []
  const strategies = candidates.value.map((candidate) => ({
    key: candidate.id,
    label: candidateLabel(candidate),
    color: colors.value.get(candidate.id)!,
    dashed: false,
    pressed: visible.value.includes(candidate.id),
    toggle: () => toggleCandidate(candidate.id),
  }))
  const references = [
    {
      key: 'asset',
      label: assetName(value),
      color: palette.value.asset,
      dashed: false,
      pressed: showAsset.value,
      toggle: () => (showAsset.value = !showAsset.value),
    },
  ]
  if (value.benchmark_symbol) {
    references.push({
      key: 'benchmark',
      label: value.benchmark_symbol,
      color: palette.value.benchmark,
      dashed: true,
      pressed: showBenchmark.value,
      toggle: () => (showBenchmark.value = !showBenchmark.value),
    })
  }
  return [...strategies, ...references]
})

const segments = computed(() => {
  const value = report.value
  if (!value) return []
  const single =
    visible.value.length === 1
      ? candidates.value.find((candidate) => candidate.id === visible.value[0])
      : undefined
  return (['training', 'test'] as const).flatMap((segment) => {
    const option = buildSegmentOption(value, segment, state.value, colors.value, palette.value)
    const reference = value.references.find((item) => item.segment === segment)
    if (!option || !reference) return []
    const metrics = single && candidateSegment(single, segment)?.metrics
    const summary = metrics
      ? [
          `Strategy ${formatPercent(metrics.total_return, { signed: true })}`,
          `${value.symbol} ${formatPercent(metrics.asset_buy_and_hold_return, { signed: true })}`,
          ...(metrics.benchmark_return !== null && value.benchmark_symbol
            ? [
                `${value.benchmark_symbol} ${formatPercent(metrics.benchmark_return, { signed: true })}`,
              ]
            : []),
        ].join(' · ')
      : `${visible.value.length} strategies`
    return [
      {
        key: segment,
        title: segment === 'training' ? 'In-sample' : 'Out-of-sample',
        option,
        summary,
      },
    ]
  })
})

const hint = computed(() =>
  tagsVisible(state.value)
    ? `B and S tags mark this strategy’s trades on the ${report.value?.symbol} price line. Hover for details.`
    : `B and S tags appear with one strategy and ${report.value ? assetName(report.value) : 'buy and hold'} shown.`,
)

const dataLine = computed(() => {
  const value = report.value
  if (!value?.data_start || !value.data_end) return ''
  const daily = value.bar_interval === '1d'
  const range = daily
    ? `${value.data_start.slice(0, 10)} → ${value.data_end.slice(0, 10)}`
    : `${minuteTimestamp(value.data_start)} → ${minuteTimestamp(value.data_end)} ET`
  return [
    daily ? 'Daily adjusted bars' : '1-minute regular-session bars',
    range,
    value.benchmark_symbol && `benchmark ${value.benchmark_symbol}`,
    'yfinance',
  ]
    .filter(Boolean)
    .join(' · ')
})
</script>

<template>
  <main class="mx-auto flex max-w-(--container-app) flex-col gap-10 px-6 pt-8 pb-20">
    <Alert v-if="query.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>{{ errorMessage(query.error.value) }}</AlertDescription>
    </Alert>
    <p v-else-if="!report" class="text-muted-foreground">Loading report…</p>

    <template v-if="report">
      <section class="flex animate-rise flex-col gap-3">
        <RouterLink
          :to="{ name: 'task', params: { threadId } }"
          class="inline-flex w-fit items-center gap-1.5 text-[13px] text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft class="size-3.5" /> Task
        </RouterLink>
        <h1 class="text-3xl font-semibold tracking-tight">{{ report.symbol }} strategy research</h1>
        <p class="max-w-190 text-zinc-700 dark:text-zinc-300">{{ report.research_objective }}</p>
        <p class="text-[13px] text-faint">{{ dataLine }}</p>
      </section>

      <section
        v-if="verdict"
        class="flex animate-rise flex-wrap gap-5 rounded-xl p-5 [animation-delay:60ms]"
        :class="{
          'bg-success-soft text-success-ink': verdict.tone === 'success',
          'bg-destructive-soft text-destructive-ink': verdict.tone === 'danger',
          'border border-border bg-surface': verdict.tone === 'neutral',
        }"
      >
        <div class="flex flex-[1_1_360px] items-start gap-3.5">
          <span
            aria-hidden="true"
            class="flex size-8 shrink-0 items-center justify-center rounded-full"
            :class="
              verdict.tone === 'success'
                ? 'bg-success text-white dark:text-zinc-950'
                : 'bg-muted text-muted-foreground'
            "
          >
            <Check v-if="verdict.tone === 'success'" class="size-4" :stroke-width="2.5" />
            <Minus v-else class="size-4" :stroke-width="2.5" />
          </span>
          <div class="flex flex-col gap-1">
            <span class="text-base font-semibold">{{ verdict.title }}</span>
            <span v-if="verdict.body">{{ verdict.body }}</span>
          </div>
        </div>
        <div v-if="comparison.length" class="flex flex-[1_1_320px] flex-col justify-center gap-2">
          <div
            v-for="row in comparison"
            :key="row.label"
            class="grid grid-cols-[150px_minmax(0,1fr)_64px] items-center gap-2.5 text-[13px]"
          >
            <span class="truncate">{{ row.label }}</span>
            <span aria-hidden="true" class="h-2 rounded-full bg-current/10">
              <span
                class="block h-full origin-left animate-grow rounded-full"
                :style="{ width: row.width, background: row.color, animationDelay: row.delay }"
              />
            </span>
            <span class="text-right font-semibold tabular-nums">{{ row.text }}</span>
          </div>
        </div>
      </section>

      <DeployStrategy
        v-if="deployable"
        :thread-id="threadId"
        class="animate-rise [animation-delay:90ms]"
      />

      <Alert v-if="report.research_incomplete" variant="warning">
        <CircleAlert />
        <AlertDescription>
          Out-of-sample validation was not completed; in-sample results are shown for audit only.
        </AlertDescription>
      </Alert>

      <section
        v-if="stats.length"
        aria-labelledby="results"
        class="flex animate-rise flex-col gap-3 [animation-delay:120ms]"
      >
        <h2 id="results" class="text-base font-semibold">
          {{ primaryTest ? 'Out-of-sample result' : 'In-sample result' }}
        </h2>
        <dl
          class="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] overflow-hidden rounded-xl border border-border"
        >
          <div
            v-for="stat in stats"
            :key="stat.label"
            class="flex flex-col gap-1 border-r border-hairline px-4.5 py-4 last:border-0"
          >
            <dt class="text-[13px] text-muted-foreground">{{ stat.label }}</dt>
            <dd class="text-2xl font-semibold tracking-tight tabular-nums">{{ stat.value }}</dd>
          </div>
        </dl>
      </section>

      <section
        aria-labelledby="curves"
        class="flex animate-rise flex-col gap-3.5 [animation-delay:180ms]"
      >
        <div class="flex flex-wrap items-center justify-between gap-3">
          <h2 id="curves" class="text-base font-semibold">Curves</h2>
          <ToggleGroupRoot
            v-model="metric"
            type="single"
            aria-label="Metric"
            class="inline-flex rounded-lg bg-muted p-0.75"
          >
            <ToggleGroupItem
              v-for="option in ['equity', 'drawdown'] as const"
              :key="option"
              :value="option"
              class="h-7 cursor-pointer rounded-md px-3 text-[13px] font-medium text-muted-foreground capitalize transition-colors duration-150 data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-[0_1px_2px_rgb(0_0_0/0.08)]"
            >
              {{ option }}
            </ToggleGroupItem>
          </ToggleGroupRoot>
        </div>
        <div role="group" aria-label="Series" class="flex flex-wrap gap-2">
          <button
            v-for="item in legend"
            :key="item.key"
            type="button"
            :aria-pressed="item.pressed"
            class="inline-flex h-7.5 cursor-pointer items-center gap-2 rounded-full border px-2.5 text-[13px] transition-colors duration-150 hover:border-input"
            :class="
              item.pressed ? 'bg-background text-foreground' : 'border-border bg-surface text-faint'
            "
            :style="item.pressed ? { borderColor: item.color } : undefined"
            @click="item.toggle"
          >
            <span
              aria-hidden="true"
              class="w-3.5 border-t-2"
              :class="item.dashed && 'border-dashed'"
              :style="{ borderColor: item.color }"
            />
            {{ item.label }}
          </button>
        </div>
        <p class="text-[13px] text-faint">{{ hint }}</p>
        <div class="grid grid-cols-[repeat(auto-fit,minmax(420px,1fr))] gap-4">
          <SegmentChart
            v-for="segment in segments"
            :key="segment.key"
            :title="segment.title"
            :summary="segment.summary"
            :option="segment.option"
          />
        </div>
      </section>

      <section
        aria-labelledby="candidates"
        class="flex animate-rise flex-col gap-3 [animation-delay:240ms]"
      >
        <div class="flex flex-col gap-0.5">
          <h2 id="candidates" class="text-base font-semibold">All candidates</h2>
          <p class="text-muted-foreground">
            Chosen on in-sample evidence, then evaluated once on the frozen out-of-sample period.
          </p>
        </div>
        <CandidateTable
          :candidates="candidates"
          :colors="colors"
          :selected-id="report.selected_candidate_id"
        />
        <p class="text-[13px] text-faint">
          Out-of-sample columns unless noted. Every segment starts from
          {{
            report.initial_capital.toLocaleString('en-US', {
              style: 'currency',
              currency: 'USD',
              maximumFractionDigits: 0,
            })
          }}.
        </p>
      </section>

      <div class="flex animate-rise flex-wrap items-start gap-8 [animation-delay:300ms]">
        <section aria-labelledby="critic" class="flex min-w-0 flex-[999_1_560px] flex-col gap-3">
          <h2 id="critic" class="text-base font-semibold">Critic review</h2>
          <p v-if="!report.critic_reviews.length" class="text-muted-foreground">
            The Critic review did not finish, so no verdict is shown.
          </p>
          <article
            v-for="(review, index) in report.critic_reviews"
            :key="index"
            class="flex flex-col gap-2"
          >
            <Badge :variant="CRITIC_VERDICT[review.verdict].tone" size="sm">
              Round {{ index + 1 }} · {{ CRITIC_VERDICT[review.verdict].label }}
            </Badge>
            <p>{{ review.summary }}</p>
            <ul v-if="review.issues.length" class="list-disc pl-5 text-muted-foreground">
              <li v-for="issue in review.issues" :key="issue">{{ issue }}</li>
            </ul>
            <p v-if="review.required_revision" class="text-muted-foreground">
              <span class="font-medium text-foreground">Requested revision:</span>
              {{ review.required_revision }}
            </p>
          </article>
        </section>
        <section aria-labelledby="limits" class="flex min-w-0 flex-[1_1_320px] flex-col gap-3">
          <h2 id="limits" class="text-base font-semibold">Limitations</h2>
          <ul
            v-if="report.limitations.length"
            class="flex list-disc flex-col gap-1.5 pl-5 text-zinc-700 dark:text-zinc-300"
          >
            <li v-for="item in report.limitations" :key="item">{{ item }}</li>
          </ul>
          <p v-else class="text-muted-foreground">No limitations were reported.</p>
          <template v-if="report.warnings.length">
            <h3 class="mt-2 font-semibold">Warnings</h3>
            <ul class="flex list-disc flex-col gap-1.5 pl-5 text-warning-ink">
              <li v-for="item in report.warnings" :key="item">{{ item }}</li>
            </ul>
          </template>
        </section>
      </div>

      <section
        v-if="report.market_context_summary || report.market_context_sources.length"
        aria-labelledby="context"
        class="flex animate-rise flex-col gap-3 [animation-delay:340ms]"
      >
        <h2 id="context" class="text-base font-semibold">Market context</h2>
        <p v-if="report.market_context_summary" class="max-w-190">
          {{ report.market_context_summary }}
        </p>
        <ul
          v-if="report.market_context_sources.length"
          class="overflow-hidden rounded-xl border border-border"
        >
          <li
            v-for="source in report.market_context_sources"
            :key="source.id"
            class="flex flex-col gap-1 border-b border-hairline px-4 py-3 transition-colors duration-150 last:border-0 hover:bg-surface"
          >
            <div class="flex flex-wrap items-center gap-2.5">
              <Badge size="sm">{{ SOURCE_LABELS[source.source_type] }}</Badge>
              <a
                :href="source.url"
                target="_blank"
                rel="noopener noreferrer"
                class="font-medium hover:text-brand"
              >
                {{ source.title }}
              </a>
              <span class="text-[13px] text-faint"
                >{{ source.source }} · {{ source.published_at }}</span
              >
            </div>
            <p class="line-clamp-2 text-[13px] text-muted-foreground">{{ source.excerpt }}</p>
          </li>
        </ul>
      </section>

      <section
        aria-labelledby="trades"
        class="flex animate-rise flex-col gap-3 [animation-delay:380ms]"
      >
        <div class="flex flex-col gap-0.5">
          <h2 id="trades" class="text-base font-semibold">Trades</h2>
          <p class="text-muted-foreground">
            Opening a strategy here shows it alone on the curves above.
          </p>
        </div>
        <TradeLedger
          :candidates="candidates"
          :colors="colors"
          :selected-id="report.selected_candidate_id"
          :open-id="openId"
          :bar-interval="report.bar_interval"
          @toggle="toggleLedger"
        />
      </section>

      <p class="border-t border-hairline pt-4 text-[13px] text-faint">
        <template v-if="report.bar_interval === '1m'">
          Adjusted 1-minute regular-session bars, next-minute-open execution and a flat position at
          every session close; annualized metrics do not apply.
        </template>
        <template v-else
          >Next-open execution, 70/30 chronological split, 0% risk-free rate.</template
        >
        Simulated results; past performance does not predict future returns.
      </p>
    </template>
  </main>
</template>
