import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError, unwrap, watchTaskEvents } from '@/api/client'
import type { TaskStreamEvent } from '@/api/types'
import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'
import { taskView } from './fixtures'

const fetchMock = vi.fn<typeof fetch>()

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  sessionStorage.clear()
  const auth = useAuthStore(pinia)
  auth.token = null
  auth.sessionExpired = false
})
afterEach(() => vi.unstubAllGlobals())

describe('api client', () => {
  it('sends the session token and returns typed data', async () => {
    useAuthStore(pinia).token = 'session-token'
    fetchMock.mockResolvedValue(json(200, taskView()))

    const task = await unwrap(
      api.GET('/api/research/{thread_id}', { params: { path: { thread_id: 'abc' } } }),
    )

    const request = fetchMock.mock.calls[0][0] as Request
    expect(request.url).toMatch(/\/api\/research\/abc$/)
    expect(request.headers.get('Authorization')).toBe('Bearer session-token')
    expect(request.credentials).toBe('include')
    expect(task.status).toBe('running')
  })

  it('renews an expired token once and retries every rejected request', async () => {
    const auth = useAuthStore(pinia)
    auth.token = 'stale-token'
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request
      if (request.url.endsWith('/api/auth/refresh')) {
        return json(200, { access_token: 'fresh-token', token_type: 'bearer' })
      }
      return request.headers.get('Authorization') === 'Bearer fresh-token'
        ? json(200, [])
        : json(401, { detail: 'Invalid or expired access token' })
    })

    const lists = await Promise.all([
      unwrap(api.GET('/api/research')),
      unwrap(api.GET('/api/research')),
    ])

    expect(lists).toEqual([[], []])
    expect(auth.token).toBe('fresh-token')
    const refreshes = fetchMock.mock.calls.filter(([request]) =>
      (request as Request).url.endsWith('/api/auth/refresh'),
    )
    expect(refreshes).toHaveLength(1)
    expect((refreshes[0][0] as Request).headers.has('Authorization')).toBe(false)
  })

  it('ends the session when it cannot be renewed but treats a failed sign-in as a plain error', async () => {
    const auth = useAuthStore(pinia)
    fetchMock.mockResolvedValue(json(401, { detail: 'Incorrect username or password' }))
    await expect(auth.signIn('researcher', 'wrong')).rejects.toThrow(
      'Incorrect username or password',
    )
    expect(auth.sessionExpired).toBe(false)

    auth.token = 'stale-token'
    fetchMock.mockImplementation(async () => json(401, { detail: 'The session has ended' }))
    await expect(unwrap(api.GET('/api/research'))).rejects.toBeInstanceOf(ApiError)
    expect(auth.isAuthenticated).toBe(false)
    expect(auth.sessionExpired).toBe(true)
  })

  it('falls back to the status code when the server sends no detail', async () => {
    fetchMock.mockResolvedValue(json(422, { detail: [{ msg: 'too long' }] }))
    await expect(unwrap(api.POST('/api/research', { body: { request: 'x' } }))).rejects.toThrow(
      'Request failed (422).',
    )
  })

  it('parses live agent, result and error events from the stream', async () => {
    const view = taskView({ run_state: null })
    fetchMock.mockResolvedValue(
      new Response(
        [
          'event: agent\ndata: {"source":"research_agent","kind":"tool","message":"Running","tool_name":"run_backtest","agent_status":null}\n\n',
          `event: result\ndata: ${JSON.stringify(view)}\n\n`,
        ].join(''),
        { headers: { 'Content-Type': 'text/event-stream' } },
      ),
    )
    const events: TaskStreamEvent[] = []

    await watchTaskEvents('abc', new AbortController().signal, (event) => events.push(event))

    expect(events.map((event) => event.type)).toEqual(['agent', 'result'])
    expect(events[1].data).toEqual(view)
  })
})
