import { computed, toValue, type MaybeRefOrGetter } from 'vue'
import { useMutation, useQuery, useQueryCache } from '@pinia/colada'
import { useIntervalFn, useWebSocket } from '@vueuse/core'

import { api, API_ORIGIN, ApiError, unwrap } from '@/api/client'
import type {
  AccountSnapshot,
  AccountView,
  BrokerCredentials,
  Order,
  OrderRequest,
  TradingMessage,
} from '@/api/types'
import { useAuthStore } from '@/stores/auth'

// Balances and positions arrive over the WebSocket; polling only notices a dropped Gateway,
// which sends nothing.
const POLL_MS = 30_000
const accountKey = ['trading', 'account']
const ordersKey = ['trading', 'orders']
const deploymentsKey = ['trading', 'deployments']
const historyKey = ['trading', 'history']

/** The linked IBKR account kept current while the page is open, or null when none is linked. */
export const useBrokerAccount = () => {
  const account = useQuery({
    key: accountKey,
    query: ({ signal }) =>
      unwrap(api.GET('/api/trading/account', { signal })).catch((cause) => {
        if (cause instanceof ApiError && cause.status === 404) return null
        throw cause
      }),
  })
  useIntervalFn(() => void account.refetch(), POLL_MS)
  return account
}

export const useLinkAccount = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (body: BrokerCredentials) => unwrap(api.PUT('/api/trading/account', { body })),
    onSuccess: () => cache.invalidateQueries({ key: accountKey }),
  })
}

export const useUnlinkAccount = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: () => unwrap(api.DELETE('/api/trading/account')),
    onSuccess: () => cache.setQueryData(accountKey, null),
  })
}

export const useOrders = () =>
  useQuery({
    key: ordersKey,
    query: ({ signal }) => unwrap(api.GET('/api/trading/orders', { signal })),
  })

/** Place an order; a retry with the same request is matched to the order it already made. */
export const usePlaceOrder = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (body: OrderRequest) => unwrap(api.POST('/api/trading/orders', { body })),
    onSuccess: (order) => upsertOrder(cache, order),
  })
}

export const useCancelOrder = () =>
  useMutation({
    mutation: (orderId: number) =>
      unwrap(
        api.DELETE('/api/trading/orders/{order_id}', { params: { path: { order_id: orderId } } }),
      ),
  })

/** Keep orders, balances, and positions current from the server's WebSocket. */
export const useTradingFeed = () => {
  const auth = useAuthStore()
  const cache = useQueryCache()
  const url = computed(() =>
    auth.token
      ? `${API_ORIGIN.replace(/^http/, 'ws')}/api/trading/ws?token=${encodeURIComponent(auth.token)}`
      : undefined,
  )
  useWebSocket(url, {
    autoReconnect: { delay: 3000 },
    // Changes made while disconnected arrive with the refetched data.
    onConnected: () => {
      void cache.invalidateQueries({ key: ordersKey })
      void cache.invalidateQueries({ key: accountKey })
    },
    onMessage: (_socket, event) => {
      const message = JSON.parse(event.data) as TradingMessage
      if (message.type === 'order') {
        upsertOrder(cache, message.data)
        // A fill joins the history, and a strategy's shares and P&L change with its orders.
        if (message.data.status === 'filled') void cache.invalidateQueries({ key: historyKey })
        if (message.data.deployment_id !== null) {
          void cache.invalidateQueries({ key: deploymentsKey })
        }
      } else {
        cache.setQueryData<AccountView | null | undefined>(accountKey, (account) =>
          account ? applySnapshot(account, message.data) : account,
        )
      }
    },
  })
}

/** A snapshot comes only from a connected Gateway, so it also shows the account connected. */
export const applySnapshot = (account: AccountView, snapshot: AccountSnapshot): AccountView => ({
  ...account,
  ...snapshot,
  connection: { connected: true, error: null },
})

// A placed order's response can arrive after the event of its next change, so the newer one wins.
export const mergeOrder = (orders: Order[], order: Order): Order[] =>
  orders.some((known) => known.id === order.id)
    ? orders.map((known) =>
        known.id === order.id && order.updated_at >= known.updated_at ? order : known,
      )
    : [order, ...orders]

const upsertOrder = (cache: ReturnType<typeof useQueryCache>, order: Order) =>
  cache.setQueryData<Order[]>(ordersKey, (orders = []) => mergeOrder(orders, order))

export const useDeployments = () =>
  useQuery({
    key: deploymentsKey,
    query: ({ signal }) => unwrap(api.GET('/api/trading/deployments', { signal })),
  })

export const useDeploymentRuns = (deploymentId: MaybeRefOrGetter<number>) =>
  useQuery({
    key: () => [...deploymentsKey, toValue(deploymentId), 'runs'],
    query: ({ signal }) =>
      unwrap(
        api.GET('/api/trading/deployments/{deployment_id}/runs', {
          params: { path: { deployment_id: toValue(deploymentId) } },
          signal,
        }),
      ),
  })

export const useDeploy = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (body: { thread_id: string; capital: number }) =>
      unwrap(api.POST('/api/trading/deployments', { body })),
    onSuccess: () => cache.invalidateQueries({ key: deploymentsKey }),
  })
}

export type DeploymentAction =
  | { id: number; action: 'pause' | 'resume' }
  | { id: number; action: 'stop'; closePosition: boolean }

export const useDeploymentAction = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (command: DeploymentAction) => {
      const params = { path: { deployment_id: command.id } }
      if (command.action === 'stop') {
        return unwrap(
          api.POST('/api/trading/deployments/{deployment_id}/stop', {
            params,
            body: { close_position: command.closePosition },
          }),
        )
      }
      return command.action === 'pause'
        ? unwrap(api.POST('/api/trading/deployments/{deployment_id}/pause', { params }))
        : unwrap(api.POST('/api/trading/deployments/{deployment_id}/resume', { params }))
    },
    onSuccess: () => cache.invalidateQueries({ key: deploymentsKey }),
  })
}

export const useTradeHistory = () =>
  useQuery({
    key: historyKey,
    query: ({ signal }) => unwrap(api.GET('/api/trading/history', { signal })),
  })

export const useStatements = () =>
  useQuery({
    key: [...historyKey, 'statements'],
    query: ({ signal }) => unwrap(api.GET('/api/trading/statements', { signal })),
  })

/** Upload an IBKR Activity Statement CSV; its new trades join the history. */
export const useUploadStatement = () => {
  const cache = useQueryCache()
  return useMutation({
    mutation: (file: File) =>
      unwrap(
        api.POST('/api/trading/statements', {
          body: { file: file as unknown as string },
          bodySerializer: ({ file }) => {
            const form = new FormData()
            form.append('file', file)
            return form
          },
        }),
      ),
    onSuccess: () => cache.invalidateQueries({ key: historyKey }),
  })
}

/** Download a file from a short-lived object-store link the API hands out. */
const download = async (link: Promise<{ url: string }>) => {
  location.assign((await link).url)
}

export const useExportHistory = () =>
  useMutation({ mutation: () => download(unwrap(api.POST('/api/trading/exports'))) })

export const downloadStatement = (statementId: number) =>
  download(
    unwrap(
      api.GET('/api/trading/statements/{statement_id}/download', {
        params: { path: { statement_id: statementId } },
      }),
    ),
  )
