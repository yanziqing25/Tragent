<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'

import type { ResumeResearchRequest } from '@/api/generated'
import { useResearchStore } from '@/stores/research'

const route = useRoute()
const research = useResearchStore()
const answer = ref('')
const copyLabel = ref('Copy ID')
const threadId = computed(() => String(route.params.threadId))
const task = computed(() =>
  research.currentTask?.thread_id === threadId.value ? research.currentTask : null,
)
const awaitingGoal = computed(() =>
  ['awaiting_clarification', 'awaiting_confirmation'].includes(task.value?.status ?? ''),
)
const taskTitle = computed(() => {
  const symbol = task.value?.goal?.symbol
  return typeof symbol === 'string' ? `${symbol} research` : 'Research task'
})
const researchBudgetRows = computed(() =>
  budgetRows(task.value?.budget_used, task.value?.budget_remaining),
)
const criticBudgetRows = computed(() =>
  budgetRows(task.value?.critic_budget_used, task.value?.critic_budget_remaining),
)
const goalRows = computed(() => {
  const goal = task.value?.goal
  if (!goal) return []

  const rows: Array<{ label: string; value: string }> = []
  const add = (label: string, value: unknown) => {
    if (value == null) return
    rows.push({ label, value: Array.isArray(value) ? value.join(', ') : String(value) })
  }

  add('Objective', goal.objective)
  add('Symbol', goal.symbol)
  add('Interval', goal.bar_interval)
  add('Start date', goal.start_date)
  add('End date', goal.end_date)

  if (!['unsupported_request', 'unrelated_request'].includes(String(goal.request_issue))) {
    add('In-sample sessions', goal.training_sessions)
    add('Out-of-sample sessions', goal.test_sessions)
    add('Strategies', goal.allowed_strategies)
    rows.push({
      label: 'Max drawdown',
      value:
        typeof goal.max_drawdown_limit === 'number'
          ? `${Number((goal.max_drawdown_limit * 100).toFixed(6))}%`
          : 'No limit',
    })
    add('Position tranches', goal.position_tranches)
    add('Benchmark', goal.benchmark_symbol)
    const costs =
      goal.costs && typeof goal.costs === 'object'
        ? (goal.costs as Record<string, unknown>)
        : goal
    add('Transaction cost', displayBps(costs.transaction_cost_bps))
    add('Slippage', displayBps(costs.slippage_bps))
  }
  return rows
})

onMounted(() => {
  if (!task.value) research.loadTask(threadId.value).catch(() => undefined)
})

onUnmounted(() => research.cancelStream())

const resume = async (body: ResumeResearchRequest = {}) => {
  try {
    await research.resumeTask(threadId.value, body)
    if (body.answer) answer.value = ''
  } catch {
    // The store exposes the safe error on this page.
  }
}

const submitAnswer = () => {
  const normalized = answer.value.trim()
  if (normalized) resume({ answer: normalized })
}

const copyThreadId = async () => {
  try {
    await navigator.clipboard.writeText(threadId.value)
    copyLabel.value = 'Copied'
  } catch {
    copyLabel.value = 'Copy failed'
  }
}

const displayBps = (value: unknown): string | null => {
  return typeof value === 'number' ? `${value} bps` : null
}

const budgetRows = (
  used: Record<string, number> | undefined,
  remaining: Record<string, number> | undefined,
) => {
  if (!used || !remaining) return []
  return Object.entries(used).map(([name, count]) => {
    const total = count + (remaining[name] ?? 0)
    const words = name.split('_').join(' ')
    return {
      name,
      label: words.charAt(0).toUpperCase() + words.slice(1),
      count,
      total,
      percent: total === 0 ? 0 : (count / total) * 100,
    }
  })
}

const statusColor = (status: string): string => {
  if (status === 'selected') return 'success'
  if (status === 'failed' || status === 'critic_rejected') return 'error'
  if (status === 'budget_exhausted' || status === 'no_strategy') return 'warning'
  if (status.startsWith('awaiting_')) return 'info'
  return status === 'running' ? 'primary' : 'secondary'
}
</script>

<template>
  <v-container max-width="1200" class="page-shell">
    <v-btn class="mb-7" variant="text" :to="{ name: 'research' }">← Back to history</v-btn>

    <header class="d-flex flex-wrap align-start justify-space-between ga-5 mb-8">
      <div>
        <div class="page-eyebrow mb-3">Durable research task</div>
        <h1 class="section-title mb-4">{{ taskTitle }}</h1>
        <div class="d-flex flex-wrap align-center ga-2 text-caption text-medium-emphasis">
          <span class="thread-id">{{ threadId }}</span>
          <v-btn size="x-small" variant="tonal" @click="copyThreadId">{{ copyLabel }}</v-btn>
        </div>
      </div>
      <v-chip
        v-if="task"
        :color="statusColor(task.status)"
        size="large"
        variant="tonal"
      >
        {{ task.status }}
      </v-chip>
    </header>

    <v-alert v-if="research.error" class="mb-6" type="error" variant="tonal">
      {{ research.error }}
    </v-alert>
    <v-progress-linear v-if="research.loading" class="mb-6" indeterminate />

    <template v-if="task">
      <v-alert
        v-if="task.stop_reason || task.safe_error"
        class="mb-6"
        :type="task.status === 'failed' ? 'error' : 'info'"
        variant="tonal"
      >
        {{ task.stop_reason ?? task.safe_error }}
      </v-alert>

      <v-card v-if="awaitingGoal" class="prompt-card mb-7">
        <v-card-text class="pa-6 pa-sm-8">
          <div class="section-kicker mb-3">Human-in-the-loop checkpoint</div>
          <h2 class="section-title mb-2">Confirm the Research Goal</h2>
          <p class="text-medium-emphasis mb-5">
            Clarify or approve the brief before the Research Agent accesses market data.
          </p>
          <v-alert
            v-if="task.clarification_hints.length"
            class="mb-4"
            title="Goal Parser needs more information"
            type="info"
            variant="tonal"
          >
            <ul class="ml-5">
              <li v-for="hint in task.clarification_hints" :key="hint">{{ hint }}</li>
            </ul>
            <div v-if="task.clarification_example" class="mt-3">
              <strong>Example answer:</strong> {{ task.clarification_example }}
            </div>
          </v-alert>
          <v-textarea
            v-model="answer"
            label="Add or change the goal in natural language"
            :placeholder="task.clarification_example ?? undefined"
            rows="2"
            maxlength="2000"
            :disabled="research.streaming"
          />
          <div class="d-flex flex-wrap ga-2">
            <v-btn
              color="primary"
              :disabled="!answer.trim() || research.streaming"
              @click="submitAnswer"
            >
              Send answer
            </v-btn>
            <v-btn
              v-if="task.status === 'awaiting_confirmation'"
              color="primary"
              :loading="research.streaming"
              @click="resume({ action: 'approve' })"
            >
              Approve and research
            </v-btn>
            <v-btn
              variant="outlined"
              :disabled="research.streaming"
              @click="resume({ action: 'reject' })"
            >
              Reject
            </v-btn>
          </div>
        </v-card-text>
      </v-card>

      <v-card v-if="task.status === 'running' && task.can_resume" class="prompt-card mb-7">
        <v-card-text class="d-flex flex-wrap align-center justify-space-between ga-4 pa-6">
          <div>
            <div class="section-kicker mb-2">Durable checkpoint ready</div>
            <div class="font-weight-bold">The Agent can continue from its saved state.</div>
          </div>
          <v-btn color="primary" size="large" :loading="research.streaming" @click="resume()">
            Resume Agent
          </v-btn>
        </v-card-text>
      </v-card>

      <v-row align="stretch">
        <v-col v-if="goalRows.length" cols="12" lg="5">
          <v-card class="surface-card h-100">
            <v-card-text class="pa-6">
              <div class="section-kicker mb-2">Confirmed brief</div>
              <h2 class="section-title mb-4">Research Goal</h2>
            </v-card-text>
            <v-table class="goal-table" density="compact">
              <tbody>
                <tr v-for="row in goalRows" :key="row.label">
                  <th>{{ row.label }}</th>
                  <td>{{ row.value }}</td>
                </tr>
              </tbody>
            </v-table>
          </v-card>
        </v-col>

        <v-col cols="12" :lg="goalRows.length ? 7 : 12">
          <v-card class="surface-card h-100">
            <v-card-text class="pa-6">
              <div class="d-flex flex-wrap align-center justify-space-between ga-3 mb-5">
                <div>
                  <div class="section-kicker mb-2">Live checkpoint</div>
                  <h2 class="section-title">Current Agent status</h2>
                </div>
                <div class="d-flex align-center ga-2">
                  <v-chip v-if="research.streaming" color="success" variant="flat">Live</v-chip>
                  <v-chip :color="statusColor(task.status)" variant="tonal">
                    {{ task.status }}
                  </v-chip>
                </div>
              </div>

              <v-sheet class="status-panel pa-5 mb-5" rounded="xl">
                <div class="detail-label">Current state</div>
                <div class="detail-value font-weight-bold">
                  {{
                    research.streaming && research.latestAgentMessage
                      ? research.latestAgentMessage
                      : task.status
                  }}
                </div>
              </v-sheet>

              <v-row>
                <v-col cols="12" sm="6">
                  <v-sheet class="agent-panel pa-5" rounded="xl">
                    <div class="section-kicker mb-4">Research Agent</div>
                    <div class="detail-label">Active Agent</div>
                    <div class="detail-value text-capitalize">{{ task.active_agent ?? 'None' }}</div>
                    <div class="detail-label mt-4">Last Agent / Action</div>
                    <div class="detail-value">
                      <span class="text-capitalize">{{ task.last_agent ?? 'None' }}</span>
                      · {{ task.last_action ?? 'None' }}
                    </div>
                    <div class="detail-label mt-5 mb-3">Live budget</div>
                    <div v-for="row in researchBudgetRows" :key="row.name" class="mb-3">
                      <div class="d-flex justify-space-between ga-3 text-caption mb-1">
                        <span>{{ row.label }}</span>
                        <strong>{{ row.count }} / {{ row.total }}</strong>
                      </div>
                      <v-progress-linear
                        color="primary"
                        height="6"
                        :model-value="row.percent"
                        rounded
                      />
                    </div>
                  </v-sheet>
                </v-col>
                <v-col cols="12" sm="6">
                  <v-sheet class="agent-panel pa-5" rounded="xl">
                    <div class="section-kicker mb-4">Critic Agent</div>
                    <div class="detail-label">Review round</div>
                    <div class="detail-value">{{ task.critic_round }}</div>
                    <div class="detail-label mt-4">Verdict</div>
                    <div class="detail-value text-capitalize">
                      {{ task.critic_verdict ?? 'Pending' }}
                    </div>
                    <div class="detail-label mt-5 mb-3">Live budget</div>
                    <div v-for="row in criticBudgetRows" :key="row.name" class="mb-3">
                      <div class="d-flex justify-space-between ga-3 text-caption mb-1">
                        <span>{{ row.label }}</span>
                        <strong>{{ row.count }} / {{ row.total }}</strong>
                      </div>
                      <v-progress-linear
                        color="secondary"
                        height="6"
                        :model-value="row.percent"
                        rounded
                      />
                    </div>
                  </v-sheet>
                </v-col>
              </v-row>

              <v-sheet v-if="task.last_observation" class="agent-panel pa-5 mt-5" rounded="xl">
                <div class="detail-label">Last observation</div>
                <div class="detail-value">{{ task.last_observation }}</div>
              </v-sheet>
            </v-card-text>
            <v-card-actions v-if="task.report_exists" class="px-6 pb-6">
              <v-btn
                color="primary"
                size="large"
                variant="tonal"
                :to="{ name: 'research-report', params: { threadId: task.thread_id } }"
              >
                View research report
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </template>
  </v-container>
</template>
