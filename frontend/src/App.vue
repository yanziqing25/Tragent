<script setup lang="ts">
import { computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Moon, Sun } from '@lucide/vue'

import BrandMark from '@/components/BrandMark.vue'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/lib/theme'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const route = useRoute()
const router = useRouter()
const { isDark, toggle } = useTheme()

const showHeader = computed(() => route.name !== 'login')
const nav = computed(() => [
  { label: 'Account', to: { name: 'account' }, active: route.name === 'account' },
  { label: 'Strategies', to: { name: 'strategies' }, active: route.name === 'strategies' },
  { label: 'Research', to: { name: 'research' }, active: route.path.startsWith('/research') },
  { label: 'Diagnostics', to: { name: 'diagnostics' }, active: route.name === 'diagnostics' },
])

watch(
  () => auth.isAuthenticated,
  (signedIn) => {
    if (!signedIn && route.meta.requiresAuth) void router.replace({ name: 'login' })
  },
)
</script>

<template>
  <header
    v-if="showHeader"
    class="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md"
  >
    <div class="mx-auto flex h-14 max-w-(--container-app) items-center gap-6 px-6">
      <RouterLink
        :to="{ name: 'account' }"
        class="flex items-center gap-2 text-[15px] font-semibold"
      >
        <BrandMark class="size-5" />
        Tragent
      </RouterLink>
      <nav aria-label="Main" class="flex flex-1 gap-1">
        <RouterLink
          v-for="item in nav"
          :key="item.label"
          :to="item.to"
          :aria-current="item.active ? 'page' : undefined"
          class="rounded-md px-2.5 py-1.5 transition-colors duration-150 hover:bg-muted hover:text-foreground"
          :class="item.active ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground'"
        >
          {{ item.label }}
        </RouterLink>
      </nav>
      <Button
        variant="ghost"
        size="icon"
        :aria-label="isDark ? 'Switch to light theme' : 'Switch to dark theme'"
        @click="toggle()"
      >
        <Sun v-if="isDark" />
        <Moon v-else />
      </Button>
      <Button variant="ghost" size="sm" @click="auth.signOut()">Sign out</Button>
    </div>
  </header>

  <RouterView v-slot="{ Component }">
    <Transition
      mode="out-in"
      enter-active-class="transition duration-200 ease-out"
      enter-from-class="opacity-0"
      leave-active-class="transition duration-150 ease-in"
      leave-to-class="opacity-0"
    >
      <component :is="Component" :key="route.path" />
    </Transition>
  </RouterView>
</template>
