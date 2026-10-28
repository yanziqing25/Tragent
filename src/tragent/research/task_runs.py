"""Redis run markers and live event channels for queued and running research tasks.

Celery reports a queued task and an unknown task with the same PENDING state, so each task
keeps one explicit marker: ``queued`` from enqueue, ``running`` while a worker advances it.
Each user's active runs also hold slots in a sorted set scored by when their marker expires,
so a lost worker's slot frees itself instead of counting against its owner forever.
"""

from __future__ import annotations

from time import time
from typing import Literal

from redis import Redis

type RunState = Literal["queued", "running"]
type Claim = Literal["claimed", "busy", "full"]

# A queued marker outlives Redis's default one-hour redelivery of an unacknowledged run.
_QUEUED_SECONDS = 2 * 60 * 60
# A running marker is refreshed on every Agent event, so it expires only after a lost worker.
_RUNNING_SECONDS = 15 * 60

# Check and take a user's slot in one step, so simultaneous requests cannot both pass the check.
# KEYS: run marker, user's slots. ARGV: now, thread ID, marker seconds, slot limit.
_CLAIM_SCRIPT = """
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', ARGV[1])
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[4]) then return -1 end
redis.call('SET', KEYS[1], 'queued', 'EX', ARGV[3])
redis.call('ZADD', KEYS[2], tonumber(ARGV[1]) + tonumber(ARGV[3]), ARGV[2])
return 1
"""
_CLAIMS: dict[int, Claim] = {1: "claimed", 0: "busy", -1: "full"}


def run_key(thread_id: str) -> str:
    return f"tragent:run:{thread_id}"


def user_runs_key(user_id: int) -> str:
    return f"tragent:user_runs:{user_id}"


def events_channel(thread_id: str) -> str:
    return f"tragent:events:{thread_id}"


def claim_run(redis: Redis, thread_id: str, *, user_id: int, limit: int) -> Claim:
    """Mark a task queued in one of its owner's slots, unless it is busy or the slots are full."""

    result = redis.register_script(_CLAIM_SCRIPT)(
        keys=[run_key(thread_id), user_runs_key(user_id)],
        args=[time(), thread_id, _QUEUED_SECONDS, limit],
    )
    return _CLAIMS[int(result)]


def mark_running(redis: Redis, thread_id: str, *, user_id: int) -> None:
    redis.set(run_key(thread_id), "running", ex=_RUNNING_SECONDS)
    _extend_slot(redis, thread_id, user_id)


def publish(redis: Redis, thread_id: str, event: str, data: str, *, user_id: int) -> None:
    """Send one SSE-ready event to live subscribers and keep the running marker fresh."""

    redis.expire(run_key(thread_id), _RUNNING_SECONDS)
    _extend_slot(redis, thread_id, user_id)
    redis.publish(events_channel(thread_id), f"{event}\n{data}")


def finish_run(redis: Redis, thread_id: str, *, user_id: int) -> None:
    """Clear the marker and free the slot, then tell subscribers to read the final checkpoint."""

    redis.delete(run_key(thread_id))
    redis.zrem(user_runs_key(user_id), thread_id)
    redis.publish(events_channel(thread_id), "done\n{}")


def run_state(redis: Redis, thread_id: str) -> RunState | None:
    return redis.get(run_key(thread_id))  # type: ignore[return-value]


def _extend_slot(redis: Redis, thread_id: str, user_id: int) -> None:
    redis.zadd(user_runs_key(user_id), {thread_id: time() + _RUNNING_SECONDS}, xx=True)


__all__ = [
    "Claim",
    "RunState",
    "claim_run",
    "events_channel",
    "finish_run",
    "mark_running",
    "publish",
    "run_key",
    "run_state",
    "user_runs_key",
]
