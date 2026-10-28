<script setup lang="ts">
import { CircleAlert } from '@lucide/vue'

import type { Order } from '@/api/types'
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
import { isOpenOrder, ORDER_STATUS } from '@/lib/status'
import { useCancelOrder, useOrders } from '@/queries/trading'

defineProps<{ currency: string }>()

const orders = useOrders()
const cancel = useCancelOrder()

const fill = (order: Order) =>
  `${formatNumber(order.filled_quantity, 0)} / ${formatNumber(order.quantity, 0)}`
</script>

<template>
  <section class="flex flex-col gap-3">
    <h2 class="text-base font-semibold">Orders</h2>
    <Alert v-if="orders.error.value || cancel.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>
        {{ errorMessage(orders.error.value ?? cancel.error.value) }}
      </AlertDescription>
    </Alert>
    <div class="overflow-x-auto rounded-xl border border-border">
      <Table class="min-w-180">
        <TableHeader>
          <TableRow>
            <TableHead>Placed</TableHead>
            <TableHead>Order</TableHead>
            <TableHead class="text-right">Filled</TableHead>
            <TableHead class="text-right">Avg price</TableHead>
            <TableHead>Status</TableHead>
            <TableHead class="text-right"><span class="sr-only">Cancel</span></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow v-for="order in orders.data.value ?? []" :key="order.id">
            <TableCell class="whitespace-nowrap text-muted-foreground tabular-nums">
              {{ formatUpdated(order.created_at) }}
            </TableCell>
            <TableCell class="whitespace-nowrap">
              <span
                class="font-medium"
                :class="order.side === 'buy' ? 'text-success-ink' : 'text-destructive-ink'"
                >{{ order.side === 'buy' ? 'Buy' : 'Sell' }}</span
              >
              {{ formatNumber(order.quantity, 0) }}
              <span class="font-medium">{{ order.symbol }}</span>
              <Badge v-if="order.deployment_id !== null" size="sm" class="ml-1">Strategy</Badge>
              <span class="text-muted-foreground">
                {{
                  order.order_type === 'market'
                    ? 'at market'
                    : `limit ${formatMoney(order.limit_price, currency)}`
                }}
              </span>
            </TableCell>
            <TableCell class="text-right tabular-nums">{{ fill(order) }}</TableCell>
            <TableCell class="text-right tabular-nums">
              {{ formatMoney(order.avg_fill_price, currency) }}
            </TableCell>
            <TableCell>
              <div class="flex flex-col gap-0.5">
                <Badge :variant="ORDER_STATUS[order.status].tone" size="sm">
                  {{ ORDER_STATUS[order.status].label }}
                </Badge>
                <span v-if="order.error" class="max-w-60 text-xs text-muted-foreground">
                  {{ order.error }}
                </span>
              </div>
            </TableCell>
            <TableCell class="text-right">
              <Button
                v-if="isOpenOrder(order)"
                variant="ghost"
                size="sm"
                :disabled="cancel.isLoading.value"
                @click="cancel.mutate(order.id)"
              >
                Cancel
              </Button>
            </TableCell>
          </TableRow>
          <TableEmpty v-if="orders.isPending.value" :colspan="6">Loading orders…</TableEmpty>
          <TableEmpty v-else-if="!orders.data.value?.length" :colspan="6">
            No orders yet.
          </TableEmpty>
        </TableBody>
      </Table>
    </div>
  </section>
</template>
