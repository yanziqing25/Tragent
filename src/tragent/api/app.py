"""FastAPI application lifecycle.

Serve it with: ``uvicorn tragent.api.app:create_app --factory``
"""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tragent import redis_client
from tragent.api import auth, doctor
from tragent.config import Settings, load_settings
from tragent.db import build_connection_pool
from tragent.research import api as research_api
from tragent.research.persistence import setup_persistence
from tragent.trading import api as trading_api
from tragent.trading import messaging
from tragent.trading.feed import TradingFeed


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = build_connection_pool(resolved_settings)
        redis = redis_client.connect(resolved_settings)
        async_redis = redis_client.connect_async(resolved_settings)
        try:
            pool.open(wait=True, timeout=resolved_settings.postgres_pool_timeout_seconds)
            setup_persistence(pool, resolved_settings)
            messaging.ensure_topics(resolved_settings)
            producer = messaging.producer(resolved_settings)
            # A group of its own gives this process every event, starting from now.
            feed = TradingFeed(
                messaging.consumer(
                    resolved_settings,
                    f"tragent-api-{uuid4()}",
                    [messaging.ORDER_EVENTS, messaging.ACCOUNT_SNAPSHOTS],
                    latest=True,
                )
            )
            relay = asyncio.create_task(feed.relay())
            app.state.settings = resolved_settings
            app.state.database_pool = pool
            app.state.redis = redis
            app.state.async_redis = async_redis
            app.state.kafka_producer = producer
            app.state.trading_feed = feed
            try:
                yield
            finally:
                feed.stop()
                await relay
                producer.flush()
        finally:
            await async_redis.aclose()
            redis.close()
            pool.close()

    app = FastAPI(title="Tragent API", lifespan=lifespan)
    # An app served from another subdomain calls the API across origins, sending the refresh
    # cookie with the auth routes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(auth.router)
    app.include_router(doctor.router)
    app.include_router(research_api.router)
    app.include_router(trading_api.router)
    return app


__all__ = ["create_app"]
