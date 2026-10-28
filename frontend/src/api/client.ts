import createClient, { type Middleware } from 'openapi-fetch'
import { EventSourceParserStream } from 'eventsource-parser/stream'

import type { paths } from './schema'
import type { TaskStreamEvent } from './types'
import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'

const SSE_MAX_BUFFER_SIZE = 1024 * 1024
const STREAM_EVENTS = new Set(['agent', 'result', 'error'])

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// Unsent copies of authorized requests, since fetch consumes a request's body.
const retries = new WeakMap<Request, Request>()

/** Attach the session token, and renew it once and retry when the API rejects it. */
const session: Middleware = {
  onRequest({ request, schemaPath }) {
    // The auth routes use the refresh cookie, not the access token.
    const token = useAuthStore(pinia).token
    if (!token || schemaPath.startsWith('/api/auth/')) return request
    request.headers.set('Authorization', `Bearer ${token}`)
    retries.set(request, request.clone())
    return request
  },
  async onResponse({ request, response }) {
    const retry = retries.get(request)
    if (response.status !== 401 || !retry) return response
    const auth = useAuthStore(pinia)
    if (!(await auth.refresh())) return response
    retry.headers.set('Authorization', `Bearer ${auth.token}`)
    return fetch(retry)
  },
}

/** The API's origin: its own domain when deployed apart from the app, else the app's. */
export const API_ORIGIN = import.meta.env.VITE_API_BASE_URL || location.origin

// Credentials let the refresh cookie travel to an API on another subdomain of the same site;
// fetch is looked up per call so tests can stub it.
export const api = createClient<paths>({
  baseUrl: API_ORIGIN,
  credentials: 'include',
  fetch: (request) => fetch(request),
})
api.use(session)

/** Return a call's data, or throw its server-provided detail as an ApiError. */
export const unwrap = async <T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> => {
  const { data, error, response } = await call
  if (response.ok) return data as T
  const detail = (error as { detail?: unknown } | undefined)?.detail
  throw new ApiError(
    response.status,
    typeof detail === 'string' ? detail : `Request failed (${response.status}).`,
  )
}

/** Follow one task's live events until the server closes the stream. */
export const watchTaskEvents = async (
  threadId: string,
  signal: AbortSignal,
  onEvent: (event: TaskStreamEvent) => void,
): Promise<void> => {
  const body = await unwrap(
    api.GET('/api/research/{thread_id}/events', {
      params: { path: { thread_id: threadId } },
      headers: { Accept: 'text/event-stream' },
      parseAs: 'stream',
      signal,
    }),
  )
  if (!body) throw new Error('The live task stream is unavailable.')
  const messages = body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(
      new EventSourceParserStream({ maxBufferSize: SSE_MAX_BUFFER_SIZE, onError: 'terminate' }),
    )
  for await (const message of messages) {
    if (!message.event || !STREAM_EVENTS.has(message.event)) {
      throw new Error('The live task stream sent an unknown event.')
    }
    onEvent({ type: message.event, data: JSON.parse(message.data) } as TaskStreamEvent)
  }
}
