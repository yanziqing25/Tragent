<script setup lang="ts">
import type { ReportCandidate } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { candidateLabel, formatNumber, formatPercent } from '@/lib/formatters'
import { candidateSegment } from '@/lib/report'

defineProps<{
  candidates: ReportCandidate[]
  colors: Map<string, string>
  selectedId: string | null
}>()
</script>

<template>
  <div class="overflow-x-auto rounded-xl border border-border">
    <Table class="min-w-205 tabular-nums">
      <TableHeader>
        <TableRow>
          <TableHead>Strategy</TableHead>
          <TableHead class="text-right">In-sample return</TableHead>
          <TableHead class="text-right">Out-of-sample return</TableHead>
          <TableHead class="text-right">Sharpe</TableHead>
          <TableHead class="text-right">Max drawdown</TableHead>
          <TableHead class="text-right">Trades</TableHead>
          <TableHead class="text-right">Exposure</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <TableRow v-for="candidate in candidates" :key="candidate.id">
          <TableCell class="font-medium whitespace-nowrap">
            <span class="inline-flex items-center gap-2">
              <span
                aria-hidden="true"
                class="size-2 rounded-sm"
                :style="{ background: colors.get(candidate.id) }"
              />
              {{ candidateLabel(candidate) }}
              <Badge v-if="candidate.id === selectedId" variant="solid" size="sm">Selected</Badge>
            </span>
          </TableCell>
          <TableCell class="text-right">
            {{
              formatPercent(candidateSegment(candidate, 'training')?.metrics.total_return, {
                signed: true,
              })
            }}
          </TableCell>
          <TableCell class="text-right font-medium">
            {{
              formatPercent(candidateSegment(candidate, 'test')?.metrics.total_return, {
                signed: true,
              })
            }}
          </TableCell>
          <TableCell class="text-right">{{
            formatNumber(candidateSegment(candidate, 'test')?.metrics.sharpe_ratio)
          }}</TableCell>
          <TableCell class="text-right">{{
            formatPercent(candidateSegment(candidate, 'test')?.metrics.max_drawdown)
          }}</TableCell>
          <TableCell class="text-right">{{
            candidateSegment(candidate, 'test')?.metrics.trade_count ?? '—'
          }}</TableCell>
          <TableCell class="text-right">{{
            formatPercent(candidateSegment(candidate, 'test')?.mean_exposure, { digits: 1 })
          }}</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  </div>
</template>
