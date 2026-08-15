<script setup lang="ts">
import { watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const route = useRoute()
const router = useRouter()

watch(
  () => auth.isAuthenticated,
  (isAuthenticated) => {
    if (!isAuthenticated && route.meta.requiresAuth) void router.replace({ name: 'login' })
  },
)

const signOut = () => {
  auth.signOut()
}
</script>

<template>
  <v-app class="app-shell">
    <v-app-bar class="app-bar" flat>
      <v-container max-width="1200" class="d-flex align-center px-4 px-sm-6">
        <v-app-bar-title>
          <router-link class="brand" :to="auth.isAuthenticated ? { name: 'research' } : { name: 'login' }">
            <span class="brand-mark" aria-hidden="true"></span>
            <span>
              <span class="brand-name">Tragent</span>
              <span class="brand-subtitle">Agent workspace</span>
            </span>
          </router-link>
        </v-app-bar-title>
        <div v-if="auth.isAuthenticated" class="d-flex align-center ga-1">
          <v-btn variant="text" :to="{ name: 'research' }">Research</v-btn>
          <v-btn variant="text" :to="{ name: 'doctor' }">Doctor</v-btn>
          <v-btn variant="text" @click="signOut">Sign out</v-btn>
        </div>
      </v-container>
    </v-app-bar>
    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>
