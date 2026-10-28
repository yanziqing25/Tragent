<script setup lang="ts">
import { ChevronDown } from '@lucide/vue'

import type { ReportCandidate, ResearchReportView } from '@/api/types'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { candidateLabel, displayTimestamp, formatMoney } from '@/lib/formatters'

const props = defineProps<{
  candidates: ReportCandidate[]
  colors: Map<string, string>
  selectedId: string | null
  openId: string | null
  barInterval: ResearchReportView['bar_interval']
}>()
const emit = defineEmits<{ toggle: [candidateId: string] }>()

const rows = (candidate: ReportCandidate) =>
  candidate.segments.flatMap((segment) =>
    segment.trades.map((trade) => ({
      ...trade,
      segment: segment.segment === 'training' ? 'In-sample' : 'Out-of-sample',
      when:
        props.barInterval === '1d'
          ? trade.date.slice(0, 10)
          : `${displayTimestamp(trade.date, props.barInterval)} ET`,
    })),
  )
</script>

<template>
  <div class="overflow-hidden rounded-xl border border-border">
    <Collapsible
      v-for="candidate in candidates"
      :key="candidate.id"
      :open="openId === candidate.id"
      class="border-b border-hairline last:border-0"
      @update:open="emit('toggle', candidate.id)"
    >
      <CollapsibleTrigger
        class="flex w-full cursor-pointer items-center gap-2.5 px-4 py-3 text-left transition-colors duration-150 hover:bg-surface"
      >
        <span
          aria-hidden="true"
          class="size-2 rounded-sm"
          :style="{ background: colors.get(candidate.id) }"
        />
        <span class="flex-1 font-medium">
          {{ candidateLabel(candidate) }}
          <span v-if="candidate.id === selectedId" class="font-normal text-faint">· selected</span>
        </span>
        <span class="text-[13px] text-faint">{{ rows(candidate).length }} trades</span>
        <ChevronDown
          class="size-4 text-faint transition-transform duration-200"
          :class="openId === candidate.id && 'rotate-180'"
        />
      </CollapsibleTrigger>
      <CollapsibleContent class="overflow-x-auto border-t border-hairline">
        <Table v-if="rows(candidate).length" class="min-w-180 tabular-nums">
          <TableHeader>
            <TableRow>
              <TableHead>Segment</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Action</TableHead>
              <TableHead class="text-right">Price</TableHead>
              <TableHead class="text-right">Quantity</TableHead>
              <TableHead class="text-right">Cost</TableHead>
              <TableHead class="text-right">Cash after</TableHead>
              <TableHead class="text-right">Realized P/L</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow v-for="(trade, index) in rows(candidate)" :key="index">
              <TableCell class="text-muted-foreground">{{ trade.segment }}</TableCell>
              <TableCell class="whitespace-nowrap">{{ trade.when }}</TableCell>
              <TableCell>
                <span
                  class="inline-flex size-4.5 items-center justify-center rounded-sm text-[11px] font-semibold text-white"
                  :class="trade.action === 'buy' ? 'bg-green-600' : 'bg-red-600'"
                  :aria-label="trade.action === 'buy' ? 'Buy' : 'Sell'"
                >
                  {{ trade.action === 'buy' ? 'B' : 'S' }}
                </span>
              </TableCell>
              <TableCell class="text-right">{{ formatMoney(trade.price, 'USD') }}</TableCell>
              <TableCell class="text-right">{{ trade.quantity.toFixed(3) }}</TableCell>
              <TableCell class="text-right text-muted-foreground">{{
                formatMoney(trade.trading_cost, 'USD')
              }}</TableCell>
              <TableCell class="text-right">{{
                formatMoney(trade.cash_balance_after, 'USD')
              }}</TableCell>
              <TableCell
                class="text-right"
                :class="
                  trade.realized_profit === null
                    ? 'text-faint'
                    : trade.realized_profit >= 0
                      ? 'text-success'
                      : 'text-destructive'
                "
              >
                {{ formatMoney(trade.realized_profit, 'USD', true) }}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
        <p v-else class="px-4 py-5 text-center text-muted-foreground">
          No trades in either segment.
        </p>
      </CollapsibleContent>
    </Collapsible>
  </div>
</template>
