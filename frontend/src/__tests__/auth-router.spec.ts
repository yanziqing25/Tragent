import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError, login } from '@/api/api'
import { authGuard } from '@/router'
import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'

vi.mock('@/api/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/api')>()),
  login: vi.fn(),
}))

describe('session authentication and route guard', () => {
  beforeEach(() => {
    useAuthStore(pinia).signOut()
    sessionStorage.clear()
    vi.clearAllMocks()
  })

  it('keeps the token across a store reload and protects research routes', async () => {
    vi.mocked(login).mockResolvedValue({ access_token: 'session-token', token_type: 'bearer' })
    const auth = useAuthStore(pinia)

    expect(authGuard({ name: 'research', meta: { requiresAuth: true } })).toEqual({
      name: 'login',
    })

    await auth.signIn('researcher', 'correct-password')

    expect(auth.token).toBe('session-token')
    expect(sessionStorage.getItem('tragent.access-token')).toBe('session-token')
    expect(authGuard({ name: 'login', meta: {} })).toEqual({ name: 'research' })
    expect(authGuard({ name: 'research', meta: { requiresAuth: true } })).toBe(true)

    auth.$dispose()
    const restored = useAuthStore(pinia)
    expect(restored.token).toBe('session-token')

    restored.signOut()
    expect(sessionStorage.getItem('tragent.access-token')).toBeNull()
  })

  it('clears an expired session while preserving login credential errors', async () => {
    vi.mocked(login).mockResolvedValue({ access_token: 'expired-token', token_type: 'bearer' })
    const auth = useAuthStore(pinia)
    await auth.signIn('researcher', 'correct-password')

    expect(auth.handleUnauthorized(new ApiError(401, 'Incorrect username or password'))).toBe(true)
    expect(auth.isAuthenticated).toBe(false)
    expect(auth.sessionExpired).toBe(true)
    expect(sessionStorage.getItem('tragent.access-token')).toBeNull()
    expect(authGuard({ name: 'research', meta: { requiresAuth: true } })).toEqual({
      name: 'login',
    })

    expect(auth.handleUnauthorized(new ApiError(400, 'Bad request'))).toBe(false)
  })
})
