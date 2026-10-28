import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { authGuard } from '@/router'
import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'

describe('session authentication and route guard', () => {
  beforeEach(() => {
    sessionStorage.clear()
    useAuthStore(pinia).signOut()
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(JSON.stringify({ access_token: 'session-token', token_type: 'bearer' }), {
            headers: { 'Content-Type': 'application/json' },
          }),
      ),
    )
  })
  afterEach(() => vi.unstubAllGlobals())

  it('keeps the token in sessionStorage and protects signed-in routes', async () => {
    const auth = useAuthStore(pinia)
    expect(authGuard({ name: 'research', meta: { requiresAuth: true } })).toEqual({ name: 'login' })

    await auth.signIn('researcher', 'correct-password')

    expect(sessionStorage.getItem('tragent.access-token')).toBe('session-token')
    expect(authGuard({ name: 'research', meta: { requiresAuth: true } })).toBe(true)
    expect(authGuard({ name: 'login', meta: {} })).toEqual({ name: 'account' })

    auth.signOut()
    await nextTick()
    expect(sessionStorage.getItem('tragent.access-token')).toBeNull()
    expect(authGuard({ name: 'task', meta: { requiresAuth: true } })).toEqual({ name: 'login' })
  })
})
