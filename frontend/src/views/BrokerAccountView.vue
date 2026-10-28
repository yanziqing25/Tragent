<script setup lang="ts">
import { computed, ref } from 'vue'
import { CircleAlert, LoaderCircle } from '@lucide/vue'

import OrderTicket from '@/components/trading/OrderTicket.vue'
import OrdersTable from '@/components/trading/OrdersTable.vue'
import TradeHistory from '@/components/trading/TradeHistory.vue'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
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
  useBrokerAccount,
  useLinkAccount,
  useTradingFeed,
  useUnlinkAccount,
} from '@/queries/trading'

const account = useBrokerAccount()
const link = useLinkAccount()
const unlink = useUnlinkAccount()
useTradingFeed()
const editing = ref(false)
const username = ref('')
const password = ref('')

const linked = computed(() => account.data.value ?? null)
const showForm = computed(() => account.data.value === null || editing.value)
const currency = computed(() => linked.value?.balances?.currency ?? 'USD')
const balances = computed(() => {
  const values = linked.value?.balances
  return [
    { label: 'Net liquidation', value: values?.net_liquidation },
    { label: 'Cash', value: values?.cash },
    { label: 'Buying power', value: values?.buying_power },
    { label: 'Available funds', value: values?.available_funds },
  ]
})

const edit = () => {
  username.value = linked.value?.username ?? ''
  password.value = ''
  editing.value = true
}

const submit = async () => {
  if (!username.value.trim() || !password.value || link.isLoading.value) return
  const saved = await link
    .mutateAsync({ username: username.value.trim(), password: password.value })
    .then(() => true)
    .catch(() => false)
  if (!saved) return
  password.value = ''
  editing.value = false
}
</script>

<template>
  <main class="mx-auto flex max-w-(--container-app) flex-col gap-8 px-6 pt-10 pb-16">
    <div class="flex animate-rise flex-wrap items-end justify-between gap-4">
      <div class="flex flex-col gap-1">
        <h1 class="text-[26px] font-semibold tracking-tight">Account</h1>
        <p class="text-muted-foreground">
          Your IBKR paper account. The platform keeps an IB Gateway logged in for you, and the page
          updates as IBKR does.
        </p>
      </div>
      <div v-if="linked && !editing" class="flex gap-2">
        <Button variant="outline" @click="edit">Change login</Button>
        <Button variant="danger" :disabled="unlink.isLoading.value" @click="unlink.mutate()">
          Unlink
        </Button>
      </div>
    </div>

    <Alert v-if="account.error.value || unlink.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>{{
        errorMessage(account.error.value ?? unlink.error.value)
      }}</AlertDescription>
    </Alert>

    <form
      v-if="showForm"
      class="flex max-w-110 animate-rise flex-col gap-4 rounded-xl border border-border p-5"
      @submit.prevent="submit"
    >
      <div class="flex flex-col gap-1">
        <h2 class="text-base font-semibold">
          {{ linked ? 'Change the paper login' : 'Link an IBKR paper account' }}
        </h2>
        <p class="text-[13px] text-muted-foreground">
          Use the paper trading username from IBKR Client Portal. The platform logs in to IBKR
          before saving it, which takes about a minute; if two-factor approval is on, approve the
          login in IBKR Mobile within two minutes.
        </p>
      </div>
      <Alert v-if="link.error.value" variant="destructive">
        <CircleAlert />
        <AlertDescription>{{ errorMessage(link.error.value) }}</AlertDescription>
      </Alert>
      <label class="flex flex-col gap-1.5">
        <span class="font-medium">Paper username</span>
        <Input
          v-model="username"
          name="ibkr-username"
          autocomplete="off"
          required
          :disabled="link.isLoading.value"
        />
      </label>
      <label class="flex flex-col gap-1.5">
        <span class="font-medium">Password</span>
        <Input
          v-model="password"
          name="ibkr-password"
          type="password"
          autocomplete="new-password"
          required
          :disabled="link.isLoading.value"
        />
      </label>
      <div class="flex justify-end gap-2">
        <Button
          v-if="linked"
          type="button"
          variant="ghost"
          :disabled="link.isLoading.value"
          @click="editing = false"
        >
          Cancel
        </Button>
        <Button type="submit" :disabled="link.isLoading.value">
          <LoaderCircle v-if="link.isLoading.value" class="animate-spin" />
          {{ link.isLoading.value ? 'Logging in to IBKR…' : 'Log in and save' }}
        </Button>
      </div>
    </form>

    <template v-if="linked && !editing">
      <section class="flex animate-rise flex-col gap-4 [animation-delay:80ms]">
        <div class="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <Badge :variant="linked.connection.connected ? 'success' : 'warning'">
            <span
              aria-hidden="true"
              class="size-1.5 rounded-full"
              :class="linked.connection.connected ? 'bg-success' : 'bg-warning'"
            />
            {{ linked.connection.connected ? 'Connected' : 'Not connected' }}
          </Badge>
          <span class="font-medium">{{ linked.username }}</span>
          <span v-if="linked.balances" class="font-mono text-[13px] text-muted-foreground">
            {{ linked.balances.account_id }}
          </span>
          <span class="text-[13px] text-faint">
            {{ linked.synced_at ? `Synced ${formatUpdated(linked.synced_at)}` : 'Not synced yet' }}
          </span>
        </div>
        <p v-if="linked.connection.error" class="text-[13px] text-muted-foreground">
          {{ linked.connection.error }}. A new Gateway takes a minute or two to log in.
        </p>
        <dl class="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div
            v-for="balance in balances"
            :key="balance.label"
            class="flex flex-col gap-1 rounded-xl border border-border px-4 py-3.5"
          >
            <dt class="text-[13px] text-muted-foreground">{{ balance.label }}</dt>
            <dd class="text-lg font-semibold tabular-nums">
              {{ formatMoney(balance.value, currency) }}
            </dd>
          </div>
        </dl>
      </section>

      <div
        class="grid animate-rise items-start gap-6 [animation-delay:160ms] lg:grid-cols-[20rem_minmax(0,1fr)]"
      >
        <OrderTicket />
        <section class="flex flex-col gap-3">
          <h2 class="text-base font-semibold">Positions</h2>
          <div class="overflow-x-auto rounded-xl border border-border">
            <Table class="min-w-180">
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead class="text-right">Quantity</TableHead>
                  <TableHead class="text-right">Average cost</TableHead>
                  <TableHead class="text-right">Price</TableHead>
                  <TableHead class="text-right">Market value</TableHead>
                  <TableHead class="text-right">Unrealized P&amp;L</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow v-for="position in linked.positions" :key="position.symbol">
                  <TableCell class="font-medium">{{ position.symbol }}</TableCell>
                  <TableCell class="text-right tabular-nums">
                    {{ formatNumber(position.quantity, 0) }}
                  </TableCell>
                  <TableCell class="text-right tabular-nums">
                    {{ formatMoney(position.average_cost, currency) }}
                  </TableCell>
                  <TableCell class="text-right tabular-nums">
                    {{ formatMoney(position.market_price, currency) }}
                  </TableCell>
                  <TableCell class="text-right tabular-nums">
                    {{ formatMoney(position.market_value, currency) }}
                  </TableCell>
                  <TableCell
                    class="text-right tabular-nums"
                    :class="
                      (position.unrealized_pnl ?? 0) < 0 ? 'text-destructive' : 'text-success-ink'
                    "
                  >
                    {{ formatMoney(position.unrealized_pnl, currency, true) }}
                  </TableCell>
                </TableRow>
                <TableEmpty v-if="!linked.positions.length" :colspan="6">
                  No open positions.
                </TableEmpty>
              </TableBody>
            </Table>
          </div>
        </section>
      </div>

      <OrdersTable :currency="currency" class="animate-rise [animation-delay:240ms]" />
      <TradeHistory :currency="currency" class="animate-rise [animation-delay:320ms]" />
    </template>
  </main>
</template>
