<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useClipboard } from '@vueuse/core'
import { ArrowLeft, ArrowRight, Check, CircleAlert, Copy, LoaderCircle } from '@lucide/vue'

import type { ResearchGoalView, ResumeResearchRequest, StrategyName } from '@/api/types'
import BudgetMeter from '@/components/BudgetMeter.vue'
import ProgressSteps from '@/components/ProgressSteps.vue'
import StatusBadge from '@/components/StatusBadge.vue'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { errorMessage } from '@/lib/errors'
import { formatPercent, studyTitle } from '@/lib/formatters'
import { useLiveTask, useResumeResearch } from '@/queries/research'

const MAX_ANSWER = 1000
const STRATEGY_NAMES: Record<StrategyName, string> = {
  moving_average: 'Moving average',
  rsi: 'RSI',
  macd: 'MACD',
  intraday_rsi: 'Intraday RSI baseline',
  rsi_price_exit: 'RSI · price-matched exit',
  rsi_vwap_exit: 'RSI · VWAP entry and exit',
}

const route = useRoute()
const threadId = computed(() => String(route.params.threadId))
const { task: query, latestMessage, streamError } = useLiveTask(threadId)
const resume = useResumeResearch(threadId)
const { copy, copied } = useClipboard({ copiedDuring: 1600 })
const answer = ref('')

const task = computed(() => query.data.value)
const awaiting = computed(
  () =>
    task.value?.status === 'awaiting_clarification' ||
    task.value?.status === 'awaiting_confirmation',
)
const running = computed(() => task.value?.status === 'running')
const title = computed(() => {
  const goal = task.value?.goal
  return studyTitle(
    goal?.symbol ?? null,
    goal?.bar_interval ?? null,
    goal?.allowed_strategies ?? null,
  )
})

const goalRows = computed(() => {
  const goal = task.value?.goal
  if (!goal) return []
  const defaulted = new Set(task.value?.defaulted_fields)
  const rows: { label: string; value: string; defaulted: boolean }[] = []
  const add = (label: string, value: string | null, ...fields: (keyof ResearchGoalView)[]) => {
    if (value !== null)
      rows.push({ label, value, defaulted: fields.some((field) => defaulted.has(field)) })
  }
  add('Symbol', goal.symbol, 'symbol')
  add(
    'Bars',
    goal.bar_interval && (goal.bar_interval === '1d' ? 'Daily' : '1-minute'),
    'bar_interval',
  )
  if (goal.bar_interval === '1m') {
    const sessions =
      goal.training_sessions !== null && goal.test_sessions !== null
        ? `${goal.training_sessions} in-sample + ${goal.test_sessions} out-of-sample`
        : null
    add('Sessions', sessions, 'training_sessions', 'test_sessions')
    add('Cutoff', goal.end_date, 'end_date')
  } else {
    add(
      'Period',
      goal.start_date && goal.end_date && `${goal.start_date} → ${goal.end_date}`,
      'start_date',
      'end_date',
    )
  }
  add(
    'Strategies',
    goal.allowed_strategies &&
      goal.allowed_strategies.map((name) => STRATEGY_NAMES[name]).join(', '),
    'allowed_strategies',
  )
  add('Benchmark', goal.benchmark_symbol, 'benchmark_symbol')
  add(
    'Max drawdown',
    goal.max_drawdown_limit === null
      ? 'No limit'
      : formatPercent(goal.max_drawdown_limit, { digits: 1 }),
    'max_drawdown_limit',
  )
  add(
    'Position tranches',
    goal.position_tranches === null ? null : String(goal.position_tranches),
    'position_tranches',
  )
  add(
    'Transaction cost',
    goal.transaction_cost_bps === null ? null : `${goal.transaction_cost_bps} bps`,
    'transaction_cost_bps',
  )
  add('Slippage', goal.slippage_bps === null ? null : `${goal.slippage_bps} bps`, 'slippage_bps')
  return rows
})

const send = async (body: ResumeResearchRequest) => {
  const updated = await resume.mutateAsync(body).catch(() => null)
  if (updated && body.answer) answer.value = ''
}
const sendAnswer = () => {
  const text = answer.value.trim()
  if (text) void send({ answer: text })
}
</script>

<template>
  <main class="mx-auto flex max-w-(--container-app) flex-col gap-7 px-6 pt-8 pb-16">
    <div class="flex animate-rise flex-col gap-3">
      <RouterLink
        :to="{ name: 'research' }"
        class="inline-flex w-fit items-center gap-1.5 text-[13px] text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft class="size-3.5" /> All research
      </RouterLink>
      <div class="flex flex-wrap items-center gap-3">
        <h1 class="text-[26px] font-semibold tracking-tight">{{ title }}</h1>
        <StatusBadge v-if="task" :task="task" />
      </div>
      <div class="flex items-center gap-1 text-[13px] text-faint">
        <span class="font-mono">{{ threadId }}</span>
        <Button variant="ghost" size="icon-sm" aria-label="Copy task ID" @click="copy(threadId)">
          <Check v-if="copied" class="text-success" />
          <Copy v-else />
        </Button>
      </div>
    </div>

    <Alert v-if="query.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>{{ errorMessage(query.error.value) }}</AlertDescription>
    </Alert>

    <template v-if="task">
      <ProgressSteps :task="task" class="animate-rise [animation-delay:60ms]" />

      <Alert v-if="streamError" variant="warning">
        <CircleAlert />
        <AlertDescription
          >Live updates stopped: {{ streamError }} Reload the page to reconnect.</AlertDescription
        >
      </Alert>
      <Alert v-if="resume.error.value" variant="destructive">
        <CircleAlert />
        <AlertDescription>{{ errorMessage(resume.error.value) }}</AlertDescription>
      </Alert>

      <div class="flex animate-rise flex-wrap items-start gap-6 [animation-delay:120ms]">
        <div class="flex min-w-0 flex-[999_1_560px] flex-col gap-4">
          <section
            v-if="awaiting"
            aria-labelledby="confirm"
            class="flex flex-col gap-4 rounded-xl border border-border p-5 shadow-[0_1px_2px_rgb(0_0_0/0.04)]"
          >
            <div class="flex flex-col gap-1">
              <h2 id="confirm" class="text-base font-semibold">Confirm the research goal</h2>
              <p class="text-muted-foreground">
                Approve to start. To change anything, answer below and the goal is parsed again.
              </p>
            </div>
            <Alert v-if="task.safe_error" variant="warning">
              <CircleAlert />
              <AlertDescription>{{ task.safe_error }}</AlertDescription>
            </Alert>
            <dl class="border-t border-hairline">
              <div
                v-for="row in goalRows"
                :key="row.label"
                class="-mx-2 grid grid-cols-[180px_minmax(0,1fr)] rounded-md border-b border-hairline px-2 transition-colors duration-150 hover:bg-surface"
              >
                <dt class="py-2 text-muted-foreground">{{ row.label }}</dt>
                <dd class="flex items-center gap-2 py-2 tabular-nums">
                  {{ row.value }}
                  <span
                    v-if="row.defaulted"
                    class="rounded bg-muted px-1.5 text-xs text-muted-foreground"
                    >default</span
                  >
                </dd>
              </div>
            </dl>
            <div
              v-if="task.clarification_hints.length"
              class="rounded-lg bg-warning-soft px-4 py-3 text-warning-ink"
            >
              <p class="font-medium">The Goal Parser needs:</p>
              <ul class="mt-1 list-disc pl-5">
                <li v-for="hint in task.clarification_hints" :key="hint">{{ hint }}</li>
              </ul>
              <p v-if="task.clarification_example" class="mt-2 text-[13px]">
                For example: {{ task.clarification_example }}
              </p>
            </div>
            <label class="flex flex-col gap-1.5">
              <span class="font-medium">
                {{ task.status === 'awaiting_clarification' ? 'Your answer' : 'Change something' }}
                <span v-if="task.status === 'awaiting_confirmation'" class="font-normal text-faint"
                  >(optional)</span
                >
              </span>
              <Textarea
                v-model="answer"
                :rows="2"
                :maxlength="MAX_ANSWER"
                :disabled="resume.isLoading.value"
                class="min-h-18"
                placeholder="e.g. Use 3 position tranches and 2 bps slippage."
                @keydown.enter.ctrl.prevent="sendAnswer"
                @keydown.enter.meta.prevent="sendAnswer"
              />
            </label>
            <div class="flex flex-wrap gap-2">
              <Button
                v-if="task.status === 'awaiting_confirmation'"
                :disabled="resume.isLoading.value"
                @click="send({ action: 'approve' })"
              >
                <LoaderCircle v-if="resume.isLoading.value" class="animate-spin" />
                Approve and start <ArrowRight />
              </Button>
              <Button
                :variant="task.status === 'awaiting_confirmation' ? 'outline' : 'default'"
                :disabled="!answer.trim() || resume.isLoading.value"
                @click="sendAnswer"
              >
                Send answer
              </Button>
              <span class="flex-1" />
              <Button
                variant="danger"
                :disabled="resume.isLoading.value"
                @click="send({ action: 'reject' })"
              >
                Reject
              </Button>
            </div>
          </section>

          <template v-else>
            <section
              v-if="running && task.run_state === 'running'"
              aria-live="polite"
              class="flex items-center gap-3.5 rounded-xl border border-info-soft bg-info-soft/40 px-5 py-4"
            >
              <span aria-hidden="true" class="relative size-2.5 shrink-0">
                <span class="absolute inset-0 animate-ring rounded-full bg-brand" />
                <span class="absolute inset-0 rounded-full bg-brand" />
              </span>
              <span class="flex-1">{{ latestMessage ?? 'The agents are working…' }}</span>
              <span class="text-xs font-medium tracking-wide text-info-ink">LIVE</span>
            </section>
            <section
              v-else-if="running && task.run_state === 'queued'"
              aria-live="polite"
              class="flex items-center gap-3.5 rounded-xl border border-border bg-surface px-5 py-4"
            >
              <span
                aria-hidden="true"
                class="size-2.5 shrink-0 animate-breathe rounded-full bg-faint"
              />
              Waiting for a free worker. You can close this page; the task starts on its own.
            </section>
            <section
              v-else-if="running && task.can_resume"
              class="flex flex-wrap items-center gap-3 rounded-xl bg-warning-soft px-5 py-4"
            >
              <div class="flex flex-[1_1_320px] flex-col gap-0.5 text-warning-ink">
                <span class="font-medium">Stopped at a checkpoint</span>
                <span>{{ task.safe_error ?? 'Completed actions and budgets are saved.' }}</span>
              </div>
              <Button :disabled="resume.isLoading.value" @click="send({})">
                <LoaderCircle v-if="resume.isLoading.value" class="animate-spin" />
                Resume
              </Button>
            </section>
            <section
              v-else-if="!running"
              class="flex flex-wrap items-center gap-3 rounded-xl px-5 py-4"
              :class="
                task.status === 'selected'
                  ? 'bg-success-soft text-success-ink'
                  : 'border border-border bg-surface'
              "
            >
              <p class="flex-[1_1_320px]">
                {{
                  task.status === 'selected'
                    ? 'The Critic accepted the selected strategy.'
                    : (task.stop_reason ??
                      task.safe_error ??
                      'The goal was rejected; nothing was run.')
                }}
              </p>
              <Button v-if="task.report_exists" as-child>
                <RouterLink :to="{ name: 'report', params: { threadId } }">
                  View report <ArrowRight />
                </RouterLink>
              </Button>
            </section>

            <section
              v-if="task.last_action"
              aria-labelledby="latest"
              class="flex flex-col gap-3 rounded-xl border border-border p-5"
            >
              <h2 id="latest" class="text-base font-semibold">Latest step</h2>
              <dl class="grid grid-cols-[140px_minmax(0,1fr)] gap-y-2">
                <dt class="text-muted-foreground">Action</dt>
                <dd>
                  <span class="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">{{
                    task.last_action
                  }}</span>
                </dd>
                <template v-if="task.last_observation">
                  <dt class="text-muted-foreground">Observation</dt>
                  <dd class="break-words">{{ task.last_observation }}</dd>
                </template>
              </dl>
            </section>

            <section
              aria-labelledby="budgets"
              class="flex flex-col gap-4 rounded-xl border border-border p-5"
            >
              <h2 id="budgets" class="text-base font-semibold">Budgets</h2>
              <div class="grid gap-6 sm:grid-cols-2">
                <div class="flex flex-col gap-3">
                  <span class="font-medium">Research Agent</span>
                  <BudgetMeter
                    label="Agent steps"
                    :used="task.budget_used.agent_steps"
                    :remaining="task.budget_remaining.agent_steps"
                    :delay="200"
                  />
                  <BudgetMeter
                    label="Tool calls"
                    :used="task.budget_used.tool_calls"
                    :remaining="task.budget_remaining.tool_calls"
                    :delay="290"
                  />
                  <BudgetMeter
                    label="Training backtests"
                    :used="task.budget_used.training_backtests"
                    :remaining="task.budget_remaining.training_backtests"
                    :delay="380"
                  />
                </div>
                <div class="flex flex-col gap-3">
                  <span class="font-medium" :class="task.critic_round === 0 && 'text-faint'">
                    Critic Agent · round {{ task.critic_round }}
                  </span>
                  <BudgetMeter
                    label="Agent steps"
                    :used="task.critic_budget_used.agent_steps"
                    :remaining="task.critic_budget_remaining.agent_steps"
                    :delay="470"
                  />
                  <BudgetMeter
                    label="Tool calls"
                    :used="task.critic_budget_used.tool_calls"
                    :remaining="task.critic_budget_remaining.tool_calls"
                    :delay="560"
                  />
                </div>
              </div>
            </section>
          </template>
        </div>

        <aside class="flex min-w-0 flex-[1_1_280px] flex-col gap-2">
          <template v-if="awaiting">
            <h2 class="text-[13px] font-medium text-muted-foreground">Your request</h2>
            <blockquote class="rounded-xl border border-hairline bg-surface px-4 py-3.5">
              {{ task.goal?.objective }}
            </blockquote>
            <p class="text-[13px] text-faint">
              Defaults come from the server settings and can be changed in your answer.
            </p>
          </template>
          <template v-else-if="goalRows.length">
            <h2 class="text-[13px] font-medium text-muted-foreground">Confirmed goal</h2>
            <dl
              class="grid grid-cols-[112px_minmax(0,1fr)] gap-y-1.5 rounded-xl border border-hairline bg-surface px-4 py-3.5 text-[13px]"
            >
              <template v-for="row in goalRows" :key="row.label">
                <dt class="text-faint">{{ row.label }}</dt>
                <dd class="tabular-nums">{{ row.value }}</dd>
              </template>
            </dl>
          </template>
        </aside>
      </div>
    </template>
  </main>
</template>
