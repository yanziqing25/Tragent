"""Push order events and account snapshots from Kafka to each user's open WebSockets.

Each API process reads both topics whole with its own consumer group, so a user's browser gets
every message whichever process its WebSocket landed on. Each message is sent as
``{"type": "order" | "account", "data": ...}``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from confluent_kafka import Consumer
from fastapi import WebSocket

from tragent.trading import messaging

_TYPES = {messaging.ORDER_EVENTS: "order", messaging.ACCOUNT_SNAPSHOTS: "account"}


class TradingFeed:
    def __init__(self, events: Consumer) -> None:
        self._events = events
        self._sockets: defaultdict[int, set[WebSocket]] = defaultdict(set)
        self._stopped = False

    @contextmanager
    def subscribe(self, user_id: int, websocket: WebSocket) -> Iterator[None]:
        self._sockets[user_id].add(websocket)
        try:
            yield
        finally:
            self._sockets[user_id].discard(websocket)

    async def relay(self) -> None:
        try:
            while not self._stopped:
                message = await asyncio.to_thread(self._events.poll, 1.0)
                if message is None or message.error():
                    continue
                payload = (
                    f'{{"type":"{_TYPES[message.topic()]}","data":{message.value().decode()}}}'
                )
                sockets = self._sockets.get(int(message.key()), set())
                # A socket closing mid-send must not stop delivery to the rest.
                await asyncio.gather(
                    *(socket.send_text(payload) for socket in sockets), return_exceptions=True
                )
        finally:
            self._events.close()

    def stop(self) -> None:
        """End the relay after its current poll."""

        self._stopped = True


__all__ = ["TradingFeed"]
