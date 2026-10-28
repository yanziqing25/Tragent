<script setup lang="ts">
import { CircleAlert, LoaderCircle } from '@lucide/vue'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { errorMessage } from '@/lib/errors'
import { useDiagnostics } from '@/queries/research'

const checks = useDiagnostics()
</script>

<template>
  <main class="mx-auto flex max-w-(--container-app) flex-col gap-6 px-6 pt-10 pb-16">
    <div class="flex animate-rise flex-wrap items-end justify-between gap-4">
      <div class="flex flex-col gap-1">
        <h1 class="text-[26px] font-semibold tracking-tight">Diagnostics</h1>
        <p class="text-muted-foreground">
          Live checks of the services the API depends on. The model and embedding checks make one
          real request each.
        </p>
      </div>
      <Button :disabled="checks.isLoading.value" @click="checks.mutate()">
        <LoaderCircle v-if="checks.isLoading.value" class="animate-spin" />
        {{ checks.isLoading.value ? 'Checking…' : 'Run checks' }}
      </Button>
    </div>

    <Alert v-if="checks.error.value" variant="destructive">
      <CircleAlert />
      <AlertDescription>{{ errorMessage(checks.error.value) }}</AlertDescription>
    </Alert>

    <ul
      v-if="checks.data.value"
      class="animate-rise overflow-hidden rounded-xl border border-border [animation-delay:80ms]"
    >
      <li
        v-for="(check, index) in checks.data.value"
        :key="check.name"
        class="grid grid-cols-[140px_84px_minmax(0,1fr)_72px] items-center gap-4 border-b border-hairline px-5 py-3.5 transition-colors duration-150 last:border-0 hover:bg-surface"
      >
        <span class="font-medium">{{ check.name }}</span>
        <Badge :variant="check.status === 'PASS' ? 'success' : 'danger'" size="sm">
          <span
            aria-hidden="true"
            class="size-1.5 animate-pop rounded-full"
            :class="check.status === 'PASS' ? 'bg-success' : 'bg-destructive'"
            :style="{ animationDelay: `${index * 90}ms` }"
          />
          {{ check.status === 'PASS' ? 'Pass' : 'Fail' }}
        </Badge>
        <span class="min-w-0 break-words text-muted-foreground">{{ check.detail }}</span>
        <span class="text-right text-[13px] text-faint tabular-nums"
          >{{ check.duration.toFixed(2) }} s</span
        >
      </li>
    </ul>
    <p
      v-else-if="!checks.isLoading.value"
      class="rounded-xl border border-dashed border-border px-5 py-8 text-center text-muted-foreground"
    >
      Run the checks to see PostgreSQL, worker, LangSmith, model and embedding status.
    </p>
  </main>
</template>
