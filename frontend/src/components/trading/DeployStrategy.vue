<script setup lang="ts">
import { computed, ref } from 'vue'
import { CircleAlert, LoaderCircle } from '@lucide/vue'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { errorMessage } from '@/lib/errors'
import { useDeploy } from '@/queries/trading'

const props = defineProps<{ threadId: string }>()

const deploy = useDeploy()
const capital = ref('')
const amount = computed(() => Number(capital.value))

const submit = () => {
  if (amount.value > 0) deploy.mutate({ thread_id: props.threadId, capital: amount.value })
}
</script>

<template>
  <section class="flex flex-col gap-3 rounded-xl border border-border p-5">
    <div class="flex flex-col gap-1">
      <h2 class="text-base font-semibold">Deploy to paper trading</h2>
      <p class="text-[13px] text-muted-foreground">
        The selected strategy trades your IBKR paper account with this capital. It starts flat, acts
        on new signals each trading day before the open, and never touches shares you hold yourself.
      </p>
    </div>
    <Alert v-if="deploy.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>{{ errorMessage(deploy.error.value) }}</AlertDescription>
    </Alert>
    <p v-if="deploy.data.value" role="status" class="text-[13px] text-success-ink">
      Deployed. Follow it on the
      <RouterLink :to="{ name: 'strategies' }" class="font-medium underline">Strategies</RouterLink>
      page.
    </p>
    <form v-else class="flex flex-wrap items-end gap-3" @submit.prevent="submit">
      <label class="flex w-48 flex-col gap-1.5">
        <span class="font-medium">Capital (USD)</span>
        <Input v-model="capital" name="capital" inputmode="decimal" placeholder="10000" />
      </label>
      <Button type="submit" :disabled="!(amount > 0) || deploy.isLoading.value">
        <LoaderCircle v-if="deploy.isLoading.value" class="animate-spin" />
        Deploy
      </Button>
    </form>
  </section>
</template>
