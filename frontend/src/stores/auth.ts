import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import { ACCESS_TOKEN_KEY, isUnauthorizedError, login } from '@/api/api'

export const SESSION_EXPIRED_MESSAGE = 'Your session expired. Please sign in again.'

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(sessionStorage.getItem(ACCESS_TOKEN_KEY))
  const sessionExpired = ref(false)
  const isAuthenticated = computed(() => token.value !== null)

  const signIn = async (username: string, password: string) => {
    const accessToken = (await login(username, password)).access_token
    sessionStorage.setItem(ACCESS_TOKEN_KEY, accessToken)
    token.value = accessToken
    sessionExpired.value = false
  }

  const signOut = () => {
    sessionExpired.value = false
    clearToken()
  }

  const handleUnauthorized = (cause: unknown): boolean => {
    if (!isUnauthorizedError(cause)) return false
    clearToken()
    sessionExpired.value = true
    return true
  }

  const clearToken = () => {
    sessionStorage.removeItem(ACCESS_TOKEN_KEY)
    token.value = null
  }

  return { token, sessionExpired, isAuthenticated, signIn, signOut, handleUnauthorized }
})
