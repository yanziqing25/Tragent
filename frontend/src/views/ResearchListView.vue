<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight, ChartLine, CircleAlert, LoaderCircle } from '@lucide/vue'

import StatusBadge from '@/components/StatusBadge.vue'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableEmpty,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Textarea } from '@/components/ui/textarea'
import { errorMessage } from '@/lib/errors'
import { formatUpdated } from '@/lib/formatters'
import { CRITIC_VERDICT } from '@/lib/status'
import { PAGE_SIZE, useStartResearch, useTaskList } from '@/queries/research'

const MAX_REQUEST = 2000
const EXAMPLES = [
  {
    label: 'AAPL · daily RSI · H1 2024 vs SPY',
    prompt:
      'Compare RSI strategies for AAPL on daily bars from 2024-01-01 to 2024-06-30 against SPY.',
  },
  {
    label: 'MSFT · daily MACD · 2024',
    prompt:
      'Evaluate a MACD strategy for MSFT on daily bars from 2024-01-01 to 2024-12-31 against SPY with a 30% maximum drawdown.',
  },
  {
    label: 'ZS · 1-minute RSI exits · 2 + 1 sessions',
    prompt:
      'Compare the fixed intraday RSI exit policies for ZS on 1-minute bars, training on 2 sessions and testing on 1.',
  },
]

const router = useRouter()
const request = ref('')
const offset = ref(0)
const tasks = useTaskList(offset)
const start = useStartResearch()
const page = computed(() => offset.value / PAGE_SIZE + 1)
const rows = computed(() => tasks.data.value ?? [])

const submit = async () => {
  const text = request.value.trim()
  if (!text || start.isLoading.value) return
  const task = await start.mutateAsync(text).catch(() => null)
  if (!task) return
  request.value = ''
  await router.push({ name: 'task', params: { threadId: task.thread_id } })
}
</script>

<template>
  <main class="mx-auto flex max-w-(--container-app) flex-col gap-10 px-6 pt-10 pb-16">
    <section aria-labelledby="new-research" class="flex animate-rise flex-col gap-3.5">
      <div class="flex flex-col gap-1">
        <h1 id="new-research" class="text-[26px] font-semibold tracking-tight">Research</h1>
        <p class="text-muted-foreground">
          Describe one study of a US stock or ETF. You confirm the parsed goal before any data is
          loaded.
        </p>
      </div>
      <form
        class="flex flex-col gap-3 rounded-xl border border-border p-4 shadow-[0_1px_2px_rgb(0_0_0/0.04)] transition-shadow duration-200 focus-within:shadow-[0_1px_2px_rgb(0_0_0/0.04),0_8px_24px_rgb(0_0_0/0.06)]"
        @submit.prevent="submit"
      >
        <label for="request" class="font-medium">New research request</label>
        <Textarea
          id="request"
          v-model="request"
          :rows="4"
          :maxlength="MAX_REQUEST"
          :disabled="start.isLoading.value"
          class="min-h-26"
          placeholder="e.g. Compare RSI strategies for AAPL on daily bars from 2024-01-01 to 2024-06-30 against SPY, with a 20% drawdown limit."
          @keydown.enter.ctrl.prevent="submit"
          @keydown.enter.meta.prevent="submit"
        />
        <Alert v-if="start.error.value" variant="destructive">
          <CircleAlert />
          <AlertDescription>{{ errorMessage(start.error.value) }}</AlertDescription>
        </Alert>
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-[13px] text-faint">Examples</span>
          <button
            v-for="example in EXAMPLES"
            :key="example.label"
            type="button"
            class="inline-flex h-7.5 cursor-pointer items-center gap-1.5 rounded-full border border-border px-2.5 text-[13px] text-muted-foreground transition-colors duration-150 hover:border-input hover:text-foreground"
            @click="request = example.prompt"
          >
            <ChartLine class="size-3" />
            {{ example.label }}
          </button>
        </div>
        <div class="flex items-center justify-end gap-3">
          <span class="text-[13px] text-faint tabular-nums"
            >{{ request.length }} / {{ MAX_REQUEST }}</span
          >
          <span class="hidden items-center gap-1 text-xs text-faint sm:inline-flex">
            <kbd
              class="rounded border border-b-2 border-border bg-surface px-1.5 font-mono text-[11px]"
              >Ctrl</kbd
            >
            <kbd
              class="rounded border border-b-2 border-border bg-surface px-1.5 font-mono text-[11px]"
              >↵</kbd
            >
          </span>
          <Button type="submit" :disabled="!request.trim() || start.isLoading.value">
            <LoaderCircle v-if="start.isLoading.value" class="animate-spin" />
            Start research
          </Button>
        </div>
      </form>
    </section>

    <section
      aria-labelledby="history"
      class="flex animate-rise flex-col gap-3 [animation-delay:80ms]"
    >
      <div class="flex items-baseline justify-between gap-3">
        <h2 id="history" class="text-base font-semibold">History</h2>
        <span class="text-[13px] text-faint">Tasks keep running after you close this page.</span>
      </div>
      <Alert v-if="tasks.error.value" variant="destructive">
        <CircleAlert />
        <AlertDescription>{{ errorMessage(tasks.error.value) }}</AlertDescription>
      </Alert>
      <div class="overflow-x-auto rounded-xl border border-border">
        <Table class="min-w-220">
          <TableHeader>
            <TableRow>
              <TableHead>Request</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Last action</TableHead>
              <TableHead>Critic</TableHead>
              <TableHead>Updated</TableHead>
              <TableHead class="text-right"><span class="sr-only">Report</span></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow v-for="task in rows" :key="task.thread_id">
              <TableCell class="max-w-90">
                <RouterLink
                  :to="{ name: 'task', params: { threadId: task.thread_id } }"
                  class="block truncate font-medium hover:text-brand"
                >
                  {{ task.request_summary }}
                </RouterLink>
              </TableCell>
              <TableCell><StatusBadge :task="task" size="sm" /></TableCell>
              <TableCell class="font-mono text-xs whitespace-nowrap text-muted-foreground">
                {{ task.last_action ?? '—' }}
              </TableCell>
              <TableCell class="whitespace-nowrap text-muted-foreground">
                {{ task.critic_verdict ? CRITIC_VERDICT[task.critic_verdict].label : '—' }}
              </TableCell>
              <TableCell class="whitespace-nowrap text-muted-foreground tabular-nums">
                {{ formatUpdated(task.updated_at) }}
              </TableCell>
              <TableCell class="text-right whitespace-nowrap">
                <RouterLink
                  v-if="task.report_exists"
                  :to="{ name: 'report', params: { threadId: task.thread_id } }"
                  class="inline-flex items-center gap-1 font-medium hover:text-brand"
                >
                  Report <ArrowRight class="size-3" />
                </RouterLink>
              </TableCell>
            </TableRow>
            <TableEmpty v-if="tasks.isPending.value" :colspan="6">Loading tasks…</TableEmpty>
            <TableEmpty v-else-if="!rows.length" :colspan="6">
              No research yet. Start with one of the examples above.
            </TableEmpty>
          </TableBody>
        </Table>
      </div>
      <div class="flex items-center justify-end gap-2">
        <span class="mr-1 text-[13px] text-faint">Page {{ page }}</span>
        <Button
          variant="outline"
          size="sm"
          :disabled="offset === 0 || tasks.isLoading.value"
          @click="offset -= PAGE_SIZE"
        >
          Previous
        </Button>
        <Button
          variant="outline"
          size="sm"
          :disabled="rows.length < PAGE_SIZE || tasks.isLoading.value"
          @click="offset += PAGE_SIZE"
        >
          Next
        </Button>
      </div>
    </section>
  </main>
</template>
