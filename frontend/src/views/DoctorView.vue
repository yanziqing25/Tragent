<script setup lang="ts">
import { ref } from 'vue'

import { runDoctor } from '@/api/api'
import type { CheckResult } from '@/api/generated'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const results = ref<CheckResult[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

const runChecks = async () => {
  loading.value = true
  error.value = null
  try {
    results.value = await runDoctor()
  } catch (cause) {
    if (!auth.handleUnauthorized(cause)) {
      error.value = cause instanceof Error ? cause.message : 'Doctor failed.'
    }
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <v-container max-width="900" class="page-shell">
    <section class="mb-8">
      <div class="page-eyebrow mb-4">System checks</div>
      <h1 class="page-title mb-4">Doctor</h1>
      <p class="page-copy">
        Check PostgreSQL, LangSmith, and the configured LLM from the running FastAPI service.
      </p>
    </section>

    <v-card class="surface-card">
      <v-card-text class="pa-6">
        <v-btn color="primary" :loading="loading" @click="runChecks">Run checks</v-btn>
        <v-alert v-if="error" class="mt-5" type="error" variant="tonal">{{ error }}</v-alert>

        <v-table v-if="results.length" class="mt-5">
          <thead>
            <tr><th>Service</th><th>Status</th><th>Time</th><th>Detail</th></tr>
          </thead>
          <tbody>
            <tr v-for="result in results" :key="result.name">
              <td>{{ result.name }}</td>
              <td>
                <v-chip
                  :color="result.status === 'PASS' ? 'success' : 'error'"
                  size="small"
                  variant="tonal"
                >
                  {{ result.status }}
                </v-chip>
              </td>
              <td>{{ result.duration.toFixed(3) }}s</td>
              <td>{{ result.detail }}</td>
            </tr>
          </tbody>
        </v-table>
      </v-card-text>
    </v-card>
  </v-container>
</template>
