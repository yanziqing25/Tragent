"""Celery workers that run durable research tasks and each morning's strategy deployments.

Start one with: ``celery -A tragent.worker worker --concurrency=16 --loglevel=info``, and the
one scheduler of the morning runs with: ``celery -A tragent.worker beat --loglevel=info``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cache
from threading import Lock
from urllib.parse import quote

from celery import Celery
from celery.schedules import crontab
from confluent_kafka import Producer
from redis import Redis

from tragent import redis_client
from tragent.config import Settings, load_settings
from tragent.db import DatabasePool, build_connection_pool
from tragent.quant.market_data import YFinanceProvider
from tragent.research.service import run_research
from tragent.trading import deployments, messaging

celery_app = Celery("tragent")
celery_app.conf.update(
    # Runs spend nearly all their time waiting on model responses, so one process runs them
    # on threads that share its connections instead of giving each run a process.
    worker_pool="threads",
    # Acknowledge after the run, so a lost worker's tasks are delivered again and resume
    # from their LangGraph checkpoints; the task lock keeps redeliveries from overlapping.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    # Deployments run before the US open, so a market order placed then fills at the open; the
    # run skips market holidays itself.
    timezone="America/New_York",
    beat_schedule={
        "run-deployments": {
            "task": "tragent.run_deployments",
            "schedule": crontab(
                hour=deployments.RUN_TIME[0],
                minute=deployments.RUN_TIME[1],
                day_of_week="mon-fri",
            ),
        }
    },
)


@celery_app.on_configure.connect
def _configure(sender: Celery, **_kwargs: object) -> None:
    sender.conf.broker_url = broker_url(load_settings())


def broker_url(settings: Settings) -> str:
    """Celery takes its broker as a URL, so build one from the Redis settings."""

    password = settings.redis_password
    auth = "" if password is None else f":{quote(password.get_secret_value(), safe='')}@"
    return f"redis://{auth}{settings.redis_host}:{settings.redis_port}/{settings.redis_database}"


@dataclass(frozen=True, slots=True)
class _Resources:
    settings: Settings
    pool: DatabasePool
    redis: Redis
    producer: Producer


_resources_lock = Lock()


def _resources() -> _Resources:
    """The worker's one database pool and Redis client, shared by all of its run threads."""

    # Runs that start together must not each open a pool on first use.
    with _resources_lock:
        return _open_resources()


@cache
def _open_resources() -> _Resources:
    settings = load_settings()
    pool = build_connection_pool(settings)
    pool.open(wait=True, timeout=settings.postgres_pool_timeout_seconds)
    return _Resources(settings, pool, redis_client.connect(settings), messaging.producer(settings))


@celery_app.task(name="tragent.run_research")
def run_research_task(thread_id: str, user_id: int, resume: dict[str, str] | None) -> None:
    resources = _resources()
    run_research(
        thread_id,
        user_id=user_id,
        resume=resume,
        settings=resources.settings,
        pool=resources.pool,
        redis=resources.redis,
    )


def enqueue_research_run(thread_id: str, user_id: int, resume: dict[str, str] | None) -> None:
    run_research_task.delay(thread_id, user_id, resume)


@celery_app.task(name="tragent.run_deployments")
def run_deployments_task() -> None:
    resources = _resources()
    session = datetime.now(deployments.EASTERN).date()
    deployments.run_all(resources.pool, resources.producer, YFinanceProvider(), session)


__all__ = [
    "broker_url",
    "celery_app",
    "enqueue_research_run",
    "run_deployments_task",
    "run_research_task",
]
