import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { useSessionStorage } from '@vueuse/core'

import { api, unwrap } from '@/api/client'

/**
 * The tab-scoped session: a short-lived bearer token kept in sessionStorage, renewed through the
 * refresh token the server keeps in an httpOnly cookie.
 */
export const useAuthStore = defineStore('auth', () => {
  const token = useSessionStorage<string | null>('tragent.access-token', null)
  const sessionExpired = ref(false)
  const isAuthenticated = computed(() => token.value !== null)
  let refreshing: Promise<boolean> | null = null

  const start = (accessToken: string) => {
    token.value = accessToken
    sessionExpired.value = false
  }

  const signIn = async (username: string, password: string) => {
    const { access_token } = await unwrap(
      api.POST('/api/auth/login', { body: { username, password } }),
    )
    start(access_token)
  }

  const register = async (username: string, password: string) => {
    const { access_token } = await unwrap(
      api.POST('/api/auth/register', { body: { username, password } }),
    )
    start(access_token)
  }

  const signOut = () => {
    token.value = null
    sessionExpired.value = false
    void api.POST('/api/auth/logout')
  }

  /** Called by the API client when the server rejects the token and it cannot be renewed. */
  const expire = () => {
    token.value = null
    sessionExpired.value = true
  }

  /** Renew the access token once, however many requests are waiting on it. */
  const refresh = () => {
    refreshing ??= unwrap(api.POST('/api/auth/refresh'))
      .then(({ access_token }) => {
        start(access_token)
        return true
      })
      .catch(() => {
        expire()
        return false
      })
      .finally(() => {
        refreshing = null
      })
    return refreshing
  }

  return { token, sessionExpired, isAuthenticated, signIn, register, signOut, expire, refresh }
})
