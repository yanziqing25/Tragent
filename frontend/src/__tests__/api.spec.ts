import { afterEach, describe, expect, it, vi } from 'vitest'

import { ACCESS_TOKEN_KEY, login, runDoctor } from '@/api/api'

describe('API authentication', () => {
  afterEach(() => {
    sessionStorage.clear()
    vi.unstubAllGlobals()
  })

  it('adds the stored JWT only to protected requests', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(okResponse({ access_token: 'jwt', token_type: 'bearer' }))
      .mockResolvedValueOnce(okResponse([]))
    vi.stubGlobal('fetch', fetchMock)
    sessionStorage.setItem(ACCESS_TOKEN_KEY, 'jwt')

    await login('researcher', 'password')
    await runDoctor()

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(new Headers(fetchMock.mock.calls[0]![1]?.headers).has('Authorization')).toBe(false)
    expect(new Headers(fetchMock.mock.calls[1]![1]?.headers).get('Authorization')).toBe(
      'Bearer jwt',
    )
  })
})

const okResponse = (body: unknown): Response => {
  return { ok: true, json: vi.fn().mockResolvedValue(body) } as unknown as Response
}
