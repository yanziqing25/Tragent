<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { useResearchStore } from '@/stores/research'

const research = useResearchStore()
const router = useRouter()
const request = ref('')
const page = computed(() => research.offset / 20 + 1)

onMounted(() => {
  research.loadTasks().catch(() => undefined)
})

onUnmounted(() => research.cancelStream())

const startTask = async () => {
  const normalized = request.value.trim()
  if (!normalized) return
  try {
    const threadId = await research.startTask(normalized)
    request.value = ''
    await router.push({ name: 'research-detail', params: { threadId } })
  } catch {
    // The store exposes the safe error beside the form.
  }
}

const loadPage = (nextOffset: number) => {
  research.loadTasks(nextOffset).catch(() => undefined)
}

const updatedAt = (value: string): string => {
  return new Date(value).toLocaleString()
}

const statusColor = (status: string): string => {
  if (status === 'selected') return 'success'
  if (status === 'failed' || status === 'critic_rejected') return 'error'
  if (status === 'budget_exhausted' || status === 'no_strategy') return 'warning'
  if (status.startsWith('awaiting_')) return 'info'
  return status === 'running' ? 'primary' : 'secondary'
}
</script>

<template>
  <v-container max-width="1200" class="page-shell">
    <section class="mb-10 mb-md-12">
      <div class="page-eyebrow mb-4">Research workspace</div>
      <h1 class="page-title mb-5">One goal. Two Agents. A durable answer.</h1>
      <p class="page-copy">
        Describe the market question in natural language. The Goal Parser clarifies the brief,
        the Research Agent investigates it, and the Critic Agent verifies the conclusion.
      </p>
    </section>

    <v-card class="prompt-card mb-9">
      <v-card-text class="pa-6 pa-sm-8">
        <div class="section-kicker mb-3">New research task</div>
        <h2 class="section-title mb-2">What should the Agent investigate?</h2>
        <p class="text-medium-emphasis mb-5">
          Use one stock symbol per task. Missing details are handled through natural-language
          clarification before research begins.
        </p>
        <v-textarea
          v-model="request"
          label="What should the Agent research?"
          rows="3"
          maxlength="2000"
          counter
          auto-grow
          :disabled="research.streaming"
        />
        <v-alert v-if="research.error" class="mb-4" type="error" variant="tonal">
          {{ research.error }}
        </v-alert>
        <v-alert
          v-if="research.streaming && research.latestAgentMessage"
          class="mb-4"
          type="info"
          variant="tonal"
        >
          {{ research.latestAgentMessage }}
        </v-alert>
        <v-btn color="primary" size="large" :loading="research.streaming" @click="startTask">
          Start Agent
        </v-btn>
      </v-card-text>
    </v-card>

    <v-card class="surface-card" :loading="research.loading">
      <v-card-text class="d-flex flex-wrap align-center justify-space-between ga-4 pa-6 pb-3">
        <div>
          <div class="section-kicker mb-2">Durable tasks</div>
          <h2 class="section-title">Research history</h2>
        </div>
        <v-chip color="primary" variant="tonal">Page {{ page }}</v-chip>
      </v-card-text>
      <v-table class="research-table">
        <thead>
          <tr>
            <th>Request</th>
            <th>Status</th>
            <th>Agent</th>
            <th>Last Action</th>
            <th>Critic</th>
            <th>Updated</th>
            <th class="text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="task in research.tasks" :key="task.thread_id">
            <td class="font-weight-medium">{{ task.request_summary }}</td>
            <td>
              <v-chip :color="statusColor(task.status)" size="small" variant="tonal">
                {{ task.status }}
              </v-chip>
            </td>
            <td class="text-capitalize">{{ task.agent ?? '—' }}</td>
            <td>{{ task.last_action ?? '—' }}</td>
            <td>
              <v-chip v-if="task.critic_verdict" color="secondary" size="small" variant="tonal">
                {{ task.critic_verdict }}
              </v-chip>
              <span v-else>—</span>
            </td>
            <td>{{ updatedAt(task.updated_at) }}</td>
            <td class="text-right text-no-wrap">
              <v-btn
                size="small"
                variant="tonal"
                :to="{ name: 'research-detail', params: { threadId: task.thread_id } }"
              >
                {{ task.can_resume ? 'Resume' : 'Details' }}
              </v-btn>
              <v-btn
                v-if="task.report_exists"
                size="small"
                variant="text"
                :to="{ name: 'research-report', params: { threadId: task.thread_id } }"
              >
                Report
              </v-btn>
            </td>
          </tr>
          <tr v-if="!research.loading && research.tasks.length === 0">
            <td colspan="7" class="text-center text-medium-emphasis py-8">No tasks yet.</td>
          </tr>
        </tbody>
      </v-table>
      <v-card-actions class="justify-end px-6 py-4">
        <v-btn
          variant="text"
          :disabled="!research.canGoBack || research.loading"
          @click="loadPage(Math.max(0, research.offset - 20))"
        >
          Previous
        </v-btn>
        <v-btn
          variant="text"
          :disabled="!research.canGoForward || research.loading"
          @click="loadPage(research.offset + 20)"
        >
          Next
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>
