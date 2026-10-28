"""Per-user run slots claimed atomically in Redis."""

from __future__ import annotations

from time import time

import fakeredis
import pytest

from tragent.research import task_runs


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def test_a_user_holds_at_most_the_limit_and_finishing_frees_a_slot(
    redis: fakeredis.FakeRedis,
) -> None:
    claims = [task_runs.claim_run(redis, f"task-{index}", user_id=1, limit=2) for index in range(3)]

    assert claims == ["claimed", "claimed", "full"]
    assert task_runs.run_state(redis, "task-2") is None
    assert task_runs.claim_run(redis, "other-user-task", user_id=2, limit=2) == "claimed"

    task_runs.finish_run(redis, "task-0", user_id=1)

    assert task_runs.claim_run(redis, "task-2", user_id=1, limit=2) == "claimed"


def test_a_queued_or_running_task_is_busy_without_taking_another_slot(
    redis: fakeredis.FakeRedis,
) -> None:
    assert task_runs.claim_run(redis, "task", user_id=1, limit=2) == "claimed"

    assert task_runs.claim_run(redis, "task", user_id=1, limit=2) == "busy"
    assert redis.zcard(task_runs.user_runs_key(1)) == 1


def test_a_lost_workers_expired_slot_is_reclaimed(redis: fakeredis.FakeRedis) -> None:
    redis.zadd(task_runs.user_runs_key(1), {"lost-task": time() - 1})

    assert task_runs.claim_run(redis, "task", user_id=1, limit=1) == "claimed"
    assert redis.zscore(task_runs.user_runs_key(1), "lost-task") is None


def test_running_progress_extends_only_a_held_slot(redis: fakeredis.FakeRedis) -> None:
    task_runs.claim_run(redis, "task", user_id=1, limit=2)
    queued_until = redis.zscore(task_runs.user_runs_key(1), "task")

    task_runs.mark_running(redis, "task", user_id=1)
    task_runs.publish(redis, "unclaimed-task", "agent", "{}", user_id=1)

    assert redis.zscore(task_runs.user_runs_key(1), "task") < queued_until
    assert redis.zscore(task_runs.user_runs_key(1), "unclaimed-task") is None
