<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { LoaderCircle } from '@lucide/vue'

import BrandMark from '@/components/BrandMark.vue'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { errorMessage } from '@/lib/errors'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const router = useRouter()
const username = ref('')
const password = ref('')
const pending = ref(false)
const error = ref<string | null>(null)
const registering = ref(false)
const title = computed(() => (registering.value ? 'Create an account' : 'Sign in'))

const switchMode = () => {
  registering.value = !registering.value
  error.value = null
}

const submit = async () => {
  if (!username.value.trim() || !password.value) return
  pending.value = true
  error.value = null
  try {
    const enter = registering.value ? auth.register : auth.signIn
    await enter(username.value.trim(), password.value)
    await router.replace({ name: 'account' })
  } catch (cause) {
    error.value = errorMessage(cause)
  } finally {
    pending.value = false
  }
}
</script>

<template>
  <main
    class="relative flex min-h-screen flex-col items-center gap-6 overflow-hidden bg-surface px-6 py-24"
  >
    <div
      aria-hidden="true"
      class="absolute inset-0 bg-[radial-gradient(var(--color-input)_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_60%_55%_at_50%_40%,#000,transparent_75%)] bg-size-[22px_22px]"
    />
    <div class="relative flex animate-rise items-center gap-2.5 text-[17px] font-semibold">
      <BrandMark class="size-6.5" />
      Tragent
    </div>

    <form
      class="relative flex w-full max-w-90 animate-rise flex-col gap-4 rounded-2xl border border-border bg-card p-6 shadow-[0_1px_2px_rgb(0_0_0/0.04),0_12px_32px_rgb(0_0_0/0.06)] [animation-delay:80ms]"
      @submit.prevent="submit"
    >
      <div class="flex flex-col gap-1">
        <h1 class="text-lg font-semibold">{{ title }}</h1>
        <p class="text-muted-foreground">
          {{
            registering
              ? 'Then link your IBKR paper account to trade.'
              : 'Use your Tragent account, not your IBKR login.'
          }}
        </p>
      </div>
      <p
        v-if="auth.sessionExpired && !error && !registering"
        role="status"
        class="rounded-lg bg-warning-soft px-3 py-2 text-[13px] text-warning-ink"
      >
        Your session expired. Sign in again to continue.
      </p>
      <p
        v-if="error"
        role="alert"
        class="rounded-lg bg-destructive-soft px-3 py-2 text-[13px] text-destructive-ink"
      >
        {{ error }}
      </p>
      <label class="flex flex-col gap-1.5">
        <span class="font-medium">Username</span>
        <Input v-model="username" name="username" autocomplete="username" required />
      </label>
      <label class="flex flex-col gap-1.5">
        <span class="font-medium">Password</span>
        <Input
          v-model="password"
          name="password"
          type="password"
          :autocomplete="registering ? 'new-password' : 'current-password'"
          required
        />
      </label>
      <Button type="submit" size="lg" :disabled="pending">
        <LoaderCircle v-if="pending" class="animate-spin" />
        {{ title }}
      </Button>
      <button
        type="button"
        class="cursor-pointer text-[13px] text-muted-foreground hover:text-foreground"
        @click="switchMode"
      >
        {{ registering ? 'Have an account? Sign in' : 'New here? Create an account' }}
      </button>
    </form>

    <p
      class="relative max-w-90 animate-rise text-center text-[13px] text-faint [animation-delay:160ms]"
    >
      Quant trading on US stocks through your own IBKR paper account, with AI research built in. Not
      financial advice.
    </p>
  </main>
</template>
