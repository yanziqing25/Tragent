<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import { SESSION_EXPIRED_MESSAGE, useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const router = useRouter()
const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref<string | null>(auth.sessionExpired ? SESSION_EXPIRED_MESSAGE : null)

const submit = async () => {
  if (!username.value.trim() || !password.value) return
  loading.value = true
  error.value = null
  try {
    await auth.signIn(username.value.trim(), password.value)
    await router.replace({ name: 'research' })
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Sign in failed.'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <v-container class="login-shell d-flex align-center py-10" max-width="1200">
    <v-row align="center" class="w-100" justify="space-between">
      <v-col cols="12" md="7">
        <div class="login-hero pr-md-10">
          <div class="page-eyebrow mb-5">Tragent · Agentic US equity research</div>
          <h1 class="login-title mb-6">Research that can explain its next move.</h1>
          <p class="page-copy mb-7">
            Turn one natural-language goal into a durable Research Agent workflow, verified by a
            separate Critic Agent before the final report.
          </p>
          <div class="d-flex flex-wrap ga-3">
            <span class="feature-pill"><span class="feature-dot"></span>Durable checkpoints</span>
            <span class="feature-pill"><span class="feature-dot"></span>Research → Critic</span>
            <span class="feature-pill"><span class="feature-dot"></span>Resume at any time</span>
          </div>
        </div>
      </v-col>
      <v-col cols="12" md="5" lg="4">
        <v-card class="login-card mt-8 mt-md-0">
          <v-card-text class="pa-7 pa-sm-9">
            <div class="section-kicker mb-3">Secure workspace</div>
            <h2 class="section-title mb-2">Welcome back</h2>
            <p class="text-medium-emphasis mb-7">Sign in to continue your Research Agent tasks.</p>
            <v-alert v-if="error" class="mb-5" type="error" variant="tonal">{{ error }}</v-alert>
            <v-form @submit.prevent="submit">
              <v-text-field
                v-model="username"
                class="mb-2"
                label="Username"
                autocomplete="username"
                required
              />
              <v-text-field
                v-model="password"
                class="mb-2"
                label="Password"
                type="password"
                autocomplete="current-password"
                required
              />
              <v-btn block color="primary" size="large" type="submit" :loading="loading">
                Enter workspace
              </v-btn>
            </v-form>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>
