"""FastAPI application lifecycle."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trading.api import auth, doctor, research
from trading.checkpoint import build_connection_pool, setup_persistence
from trading.config import Settings, load_settings

_WEB_ORIGIN = "https://trading.yzking.cn"


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_settings = settings or load_settings()
        pool = build_connection_pool(resolved_settings)
        try:
            pool.open(wait=True, timeout=resolved_settings.postgres_pool_timeout_seconds)
            setup_persistence(pool)
            app.state.settings = resolved_settings
            app.state.database_pool = pool
            yield
        finally:
            pool.close()

    app = FastAPI(title="Tragent API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_WEB_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(auth.router)
    app.include_router(doctor.router)
    app.include_router(research.router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
