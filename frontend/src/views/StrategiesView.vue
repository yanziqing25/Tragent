<script setup lang="ts">
import { ref } from 'vue'
import { ChevronDown, CircleAlert } from '@lucide/vue'

import type { DeploymentView } from '@/api/types'
import DeploymentRuns from '@/components/trading/DeploymentRuns.vue'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { errorMessage } from '@/lib/errors'
import { candidateLabel, formatMoney, formatNumber, formatUpdated } from '@/lib/formatters'
import { useDeploymentAction, useDeployments, useTradingFeed } from '@/queries/trading'

const deployments = useDeployments()
const action = useDeploymentAction()
useTradingFeed()
// The deployment whose Stop choice is showing.
const stopping = ref<number | null>(null)

const STATUS = {
  active: { label: 'Active', tone: 'success' },
  paused: { label: 'Paused', tone: 'warning' },
  stopped: { label: 'Stopped', tone: 'neutral' },
} as const

const stop = (deployment: DeploymentView, closePosition: boolean) => {
  action.mutate({ id: deployment.id, action: 'stop', closePosition })
  stopping.value = null
}
</script>

<template>
  <main class="mx-auto flex max-w-(--container-app) flex-col gap-6 px-6 pt-10 pb-16">
    <div class="flex animate-rise flex-col gap-1">
      <h1 class="text-[26px] font-semibold tracking-tight">Strategies</h1>
      <p class="text-muted-foreground">
        Strategies deployed from research run every trading day at 09:00 ET, before the open.
        <template v-if="deployments.data.value">
          Next run {{ formatUpdated(deployments.data.value.next_run_at) }}.
        </template>
      </p>
    </div>

    <Alert v-if="deployments.error.value || action.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>
        {{ errorMessage(deployments.error.value ?? action.error.value) }}
      </AlertDescription>
    </Alert>

    <p
      v-if="deployments.data.value && !deployments.data.value.deployments.length"
      class="rounded-xl border border-dashed border-border px-5 py-8 text-center text-muted-foreground"
    >
      Nothing deployed yet. Open a research report whose strategy the Critic accepted and deploy it
      from there.
    </p>

    <Collapsible
      v-for="deployment in deployments.data.value?.deployments ?? []"
      :key="deployment.id"
      v-slot="{ open }"
      class="animate-rise overflow-hidden rounded-xl border border-border"
    >
      <div class="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4">
        <div class="flex min-w-56 flex-1 flex-col gap-1">
          <div class="flex items-center gap-2">
            <span class="font-semibold">{{ deployment.symbol }}</span>
            <span class="text-muted-foreground">{{ candidateLabel(deployment) }}</span>
            <Badge :variant="STATUS[deployment.status].tone" size="sm">
              {{ STATUS[deployment.status].label }}
            </Badge>
          </div>
          <RouterLink
            :to="{ name: 'report', params: { threadId: deployment.thread_id } }"
            class="w-fit text-[13px] text-faint hover:text-foreground"
          >
            From research · deployed {{ formatUpdated(deployment.created_at) }}
          </RouterLink>
        </div>
        <dl class="flex gap-6 text-[13px]">
          <div class="flex flex-col">
            <dt class="text-muted-foreground">Capital</dt>
            <dd class="tabular-nums">{{ formatMoney(deployment.capital, 'USD') }}</dd>
          </div>
          <div class="flex flex-col">
            <dt class="text-muted-foreground">Tranches</dt>
            <dd class="tabular-nums">
              {{ deployment.tranches }} / {{ deployment.position_tranches }}
            </dd>
          </div>
          <div class="flex flex-col">
            <dt class="text-muted-foreground">Shares</dt>
            <dd class="tabular-nums">{{ formatNumber(deployment.held_shares, 0) }}</dd>
          </div>
          <div class="flex flex-col">
            <dt class="text-muted-foreground">P&amp;L</dt>
            <dd
              class="tabular-nums"
              :class="(deployment.pnl ?? 0) < 0 ? 'text-destructive' : 'text-success-ink'"
            >
              {{ formatMoney(deployment.pnl, 'USD', true) }}
            </dd>
          </div>
        </dl>
        <div class="flex items-center gap-1">
          <template v-if="stopping === deployment.id">
            <Button size="sm" variant="ghost" @click="stop(deployment, false)">Keep shares</Button>
            <Button
              size="sm"
              variant="danger"
              :disabled="!deployment.held_shares"
              @click="stop(deployment, true)"
            >
              Sell {{ formatNumber(deployment.held_shares, 0) }} shares
            </Button>
          </template>
          <template v-else-if="deployment.status !== 'stopped'">
            <Button
              size="sm"
              variant="ghost"
              :disabled="action.isLoading.value"
              @click="
                action.mutate({
                  id: deployment.id,
                  action: deployment.status === 'active' ? 'pause' : 'resume',
                })
              "
            >
              {{ deployment.status === 'active' ? 'Pause' : 'Resume' }}
            </Button>
            <Button size="sm" variant="danger" @click="stopping = deployment.id">Stop</Button>
          </template>
          <CollapsibleTrigger
            class="inline-flex size-8 cursor-pointer items-center justify-center rounded-lg text-muted-foreground hover:bg-muted"
            aria-label="Daily runs"
          >
            <ChevronDown class="size-4 transition-transform" :class="open && 'rotate-180'" />
          </CollapsibleTrigger>
        </div>
      </div>
      <CollapsibleContent class="overflow-x-auto border-t border-hairline">
        <DeploymentRuns :deployment-id="deployment.id" />
      </CollapsibleContent>
    </Collapsible>
  </main>
</template>
