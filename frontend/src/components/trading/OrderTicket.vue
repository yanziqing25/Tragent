<script setup lang="ts">
import { computed, ref } from 'vue'
import { CircleAlert, LoaderCircle } from '@lucide/vue'

import type { OrderRequest } from '@/api/types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { errorMessage } from '@/lib/errors'
import { usePlaceOrder } from '@/queries/trading'

const place = usePlaceOrder()
const symbol = ref('')
const side = ref<OrderRequest['side']>('buy')
const orderType = ref<OrderRequest['order_type']>('market')
const quantity = ref('')
const limitPrice = ref('')
// One key per order: a retry after a lost response reuses it, so the server makes one order.
const clientOrderId = ref(crypto.randomUUID())

const shares = computed(() => Number(quantity.value))
const price = computed(() => Number(limitPrice.value))
const ready = computed(
  () =>
    symbol.value.trim() !== '' &&
    Number.isInteger(shares.value) &&
    shares.value > 0 &&
    (orderType.value === 'market' || price.value > 0),
)

const choices = {
  side: [
    { value: 'buy', label: 'Buy' },
    { value: 'sell', label: 'Sell' },
  ],
  orderType: [
    { value: 'market', label: 'Market' },
    { value: 'limit', label: 'Limit' },
  ],
} as const

const submit = async () => {
  if (!ready.value || place.isLoading.value) return
  const placed = await place
    .mutateAsync({
      client_order_id: clientOrderId.value,
      symbol: symbol.value.trim(),
      side: side.value,
      order_type: orderType.value,
      quantity: shares.value,
      limit_price: orderType.value === 'limit' ? price.value : null,
    })
    .then(() => true)
    .catch(() => false)
  if (!placed) return
  clientOrderId.value = crypto.randomUUID()
  quantity.value = ''
  limitPrice.value = ''
}
</script>

<template>
  <form class="flex flex-col gap-4 rounded-xl border border-border p-5" @submit.prevent="submit">
    <div class="flex flex-col gap-1">
      <h2 class="text-base font-semibold">New order</h2>
      <p class="text-[13px] text-muted-foreground">
        US stocks routed by IBKR, good for the day. Paper account only.
      </p>
    </div>
    <Alert v-if="place.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>{{ errorMessage(place.error.value) }}</AlertDescription>
    </Alert>
    <div class="grid grid-cols-2 gap-2">
      <div class="flex rounded-lg bg-muted p-0.5" role="radiogroup" aria-label="Side">
        <button
          v-for="choice in choices.side"
          :key="choice.value"
          type="button"
          role="radio"
          :aria-checked="side === choice.value"
          class="h-8 flex-1 cursor-pointer rounded-md text-[13px] font-medium transition-colors duration-150"
          :class="
            side === choice.value
              ? choice.value === 'buy'
                ? 'bg-background text-success-ink shadow-sm'
                : 'bg-background text-destructive-ink shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          "
          @click="side = choice.value"
        >
          {{ choice.label }}
        </button>
      </div>
      <div class="flex rounded-lg bg-muted p-0.5" role="radiogroup" aria-label="Order type">
        <button
          v-for="choice in choices.orderType"
          :key="choice.value"
          type="button"
          role="radio"
          :aria-checked="orderType === choice.value"
          class="h-8 flex-1 cursor-pointer rounded-md text-[13px] font-medium transition-colors duration-150"
          :class="
            orderType === choice.value
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          "
          @click="orderType = choice.value"
        >
          {{ choice.label }}
        </button>
      </div>
    </div>
    <label class="flex flex-col gap-1.5">
      <span class="font-medium">Symbol</span>
      <Input
        v-model="symbol"
        name="symbol"
        autocomplete="off"
        placeholder="AAPL"
        class="uppercase"
      />
    </label>
    <div class="grid grid-cols-2 gap-3">
      <label class="flex flex-col gap-1.5">
        <span class="font-medium">Shares</span>
        <Input v-model="quantity" name="quantity" inputmode="numeric" placeholder="10" />
      </label>
      <label class="flex flex-col gap-1.5">
        <span class="font-medium" :class="orderType === 'market' && 'text-faint'">Limit price</span>
        <Input
          v-model="limitPrice"
          name="limit-price"
          inputmode="decimal"
          :placeholder="orderType === 'market' ? 'At market' : '150.00'"
          :disabled="orderType === 'market'"
        />
      </label>
    </div>
    <Button type="submit" :disabled="!ready || place.isLoading.value">
      <LoaderCircle v-if="place.isLoading.value" class="animate-spin" />
      {{ side === 'buy' ? 'Buy' : 'Sell' }} {{ symbol.trim().toUpperCase() || 'shares' }}
    </Button>
  </form>
</template>
