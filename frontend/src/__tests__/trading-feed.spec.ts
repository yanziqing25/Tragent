import { describe, expect, it } from 'vitest'

import type { AccountView, Order } from '@/api/types'
import { applySnapshot, mergeOrder } from '@/queries/trading'

const order = (id: number, status: Order['status'], updatedAt: string): Order => ({
  id,
  user_id: 5,
  client_order_id: '00000000-0000-4000-8000-000000000000',
  symbol: 'AAPL',
  side: 'buy',
  order_type: 'market',
  quantity: 10,
  limit_price: null,
  status,
  filled_quantity: 0,
  avg_fill_price: null,
  ib_order_id: null,
  error: null,
  deployment_id: null,
  created_at: '2026-10-23T15:00:00Z',
  updated_at: updatedAt,
})

describe('order list merging', () => {
  it('adds new orders first and keeps the newer version of a known one', () => {
    const working = order(1, 'submitted', '2026-10-23T15:00:02Z')

    const placed = mergeOrder([], order(1, 'pending_submit', '2026-10-23T15:00:00Z'))
    const updated = mergeOrder(placed, working)
    const lateResponse = mergeOrder(updated, order(1, 'pending_submit', '2026-10-23T15:00:00Z'))
    const another = mergeOrder(lateResponse, order(2, 'pending_submit', '2026-10-23T15:00:03Z'))

    expect(updated).toEqual([working])
    expect(lateResponse).toEqual([working])
    expect(another.map((item) => item.id)).toEqual([2, 1])
  })
})

describe('account snapshots', () => {
  it('replace balances and positions and show the Gateway connected', () => {
    const account: AccountView = {
      username: 'paper',
      connection: { connected: false, error: 'The broker process is not running' },
      balances: null,
      positions: [],
      synced_at: null,
    }
    const snapshot = {
      balances: {
        account_id: 'DU1',
        currency: 'USD',
        net_liquidation: 1000,
        cash: 900,
        buying_power: 4000,
        available_funds: 900,
      },
      positions: [
        {
          symbol: 'AAPL',
          quantity: 1,
          average_cost: 95,
          market_price: 100,
          market_value: 100,
          unrealized_pnl: 5,
        },
      ],
      synced_at: '2026-10-23T17:00:00Z',
    }

    expect(applySnapshot(account, snapshot)).toEqual({
      username: 'paper',
      connection: { connected: true, error: null },
      ...snapshot,
    })
  })
})
