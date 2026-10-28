<script setup lang="ts">
import {
  Table,
  TableBody,
  TableCell,
  TableEmpty,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { formatNumber } from '@/lib/formatters'
import { useDeploymentRuns } from '@/queries/trading'

const props = defineProps<{ deploymentId: number }>()

const runs = useDeploymentRuns(() => props.deploymentId)
const SIGNAL = { 1: 'Buy', [-1]: 'Sell', 0: 'None' } as Record<number, string>
</script>

<template>
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Session</TableHead>
        <TableHead>Signal</TableHead>
        <TableHead class="text-right">Tranches</TableHead>
        <TableHead class="text-right">Target shares</TableHead>
        <TableHead class="text-right">Held before</TableHead>
        <TableHead>Result</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      <TableRow v-for="run in runs.data.value ?? []" :key="run.session">
        <TableCell class="tabular-nums">{{ run.session }}</TableCell>
        <TableCell>{{ run.signal === null ? '—' : SIGNAL[run.signal] }}</TableCell>
        <TableCell class="text-right tabular-nums">{{ run.tranches }}</TableCell>
        <TableCell class="text-right tabular-nums">{{ run.target_shares ?? '—' }}</TableCell>
        <TableCell class="text-right tabular-nums">{{
          formatNumber(run.held_shares, 0)
        }}</TableCell>
        <TableCell class="text-muted-foreground">
          {{ run.note ?? (run.order_id ? `Order #${run.order_id}` : 'No change') }}
        </TableCell>
      </TableRow>
      <TableEmpty v-if="runs.isPending.value" :colspan="6">Loading runs…</TableEmpty>
      <TableEmpty v-else-if="!runs.data.value?.length" :colspan="6">
        No runs yet. The first one is the next trading morning.
      </TableEmpty>
    </TableBody>
  </Table>
</template>
