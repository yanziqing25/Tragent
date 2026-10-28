<script setup lang="ts">
import { ref } from 'vue'
import { CircleAlert, Download, LoaderCircle, Upload } from '@lucide/vue'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
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
import { errorMessage } from '@/lib/errors'
import { formatMoney, formatNumber, formatUpdated } from '@/lib/formatters'
import {
  downloadStatement,
  useExportHistory,
  useStatements,
  useTradeHistory,
  useUploadStatement,
} from '@/queries/trading'

defineProps<{ currency: string }>()

const trades = useTradeHistory()
const statements = useStatements()
const upload = useUploadStatement()
const exporting = useExportHistory()
const picker = ref<HTMLInputElement | null>(null)

const pick = (event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) upload.mutate(file)
  input.value = ''
}
</script>

<template>
  <section class="flex flex-col gap-3">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="flex flex-col gap-0.5">
        <h2 class="text-base font-semibold">Trade history</h2>
        <p class="text-[13px] text-muted-foreground">
          Fills placed here, plus trades imported from IBKR Activity Statements (CSV).
        </p>
      </div>
      <div class="flex gap-2">
        <input ref="picker" type="file" accept=".csv,text/csv" class="hidden" @change="pick" />
        <Button
          variant="outline"
          size="sm"
          :disabled="upload.isLoading.value"
          @click="picker?.click()"
        >
          <LoaderCircle v-if="upload.isLoading.value" class="animate-spin" />
          <Upload v-else />
          Import statement
        </Button>
        <Button
          variant="outline"
          size="sm"
          :disabled="exporting.isLoading.value"
          @click="exporting.mutate()"
        >
          <Download />
          Export CSV
        </Button>
      </div>
    </div>

    <Alert
      v-if="upload.error.value || exporting.error.value || trades.error.value"
      variant="destructive"
    >
      <CircleAlert />
      <AlertDescription>
        {{ errorMessage(upload.error.value ?? exporting.error.value ?? trades.error.value) }}
      </AlertDescription>
    </Alert>
    <p v-if="upload.data.value" role="status" class="text-[13px] text-success-ink">
      Imported {{ upload.data.value.imported }} trades from {{ upload.data.value.filename }}
      <template v-if="upload.data.value.skipped">
        and skipped {{ upload.data.value.skipped }} already imported
      </template>
      .
    </p>

    <ul v-if="statements.data.value?.length" class="flex flex-wrap gap-2">
      <li v-for="statement in statements.data.value" :key="statement.id">
        <button
          type="button"
          class="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-[13px] text-muted-foreground hover:text-foreground"
          :title="`Uploaded ${formatUpdated(statement.uploaded_at)}`"
          @click="downloadStatement(statement.id)"
        >
          <Download class="size-3" />
          {{ statement.filename }} · {{ statement.imported }} trades
        </button>
      </li>
    </ul>

    <div class="overflow-x-auto rounded-xl border border-border">
      <Table class="min-w-180">
        <TableHeader>
          <TableRow>
            <TableHead>Time</TableHead>
            <TableHead>Trade</TableHead>
            <TableHead class="text-right">Price</TableHead>
            <TableHead class="text-right">Commission</TableHead>
            <TableHead>Source</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow
            v-for="(trade, index) in trades.data.value ?? []"
            :key="`${trade.source}-${trade.traded_at}-${index}`"
          >
            <TableCell class="whitespace-nowrap text-muted-foreground tabular-nums">
              {{ formatUpdated(trade.traded_at) }}
            </TableCell>
            <TableCell class="whitespace-nowrap">
              <span
                class="font-medium"
                :class="trade.side === 'buy' ? 'text-success-ink' : 'text-destructive-ink'"
                >{{ trade.side === 'buy' ? 'Buy' : 'Sell' }}</span
              >
              {{ formatNumber(trade.quantity, 0) }}
              <span class="font-medium">{{ trade.symbol }}</span>
            </TableCell>
            <TableCell class="text-right tabular-nums">
              {{ formatMoney(trade.price, currency) }}
            </TableCell>
            <TableCell class="text-right tabular-nums">
              {{ formatMoney(trade.commission, currency) }}
            </TableCell>
            <TableCell>
              <Badge size="sm">{{ trade.source === 'platform' ? 'Tragent' : 'Statement' }}</Badge>
            </TableCell>
          </TableRow>
          <TableEmpty v-if="trades.isPending.value" :colspan="5">Loading trades…</TableEmpty>
          <TableEmpty v-else-if="!trades.data.value?.length" :colspan="5">
            No trades yet. Import an Activity Statement to bring in earlier trades.
          </TableEmpty>
        </TableBody>
      </Table>
    </div>
  </section>
</template>
