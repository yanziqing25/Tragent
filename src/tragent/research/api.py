"""Authenticated HTTP presentation for durable research tasks run by Celery workers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Literal, Self

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from tragent.api.auth import CurrentUserDependency
from tragent.api.dependencies import (
    AsyncRedisDependency,
    DatabasePoolDependency,
    RedisDependency,
    SettingsDependency,
)
from tragent.research import task_runs
from tragent.research.agent.models import ClarificationAnswer
from tragent.research.reporting import ResearchReportView
from tragent.research.service import (
    ReportNotFoundError,
    ResearchApplicationError,
    ResearchTaskSummary,
    ResearchTaskView,
    TaskBusyError,
    TaskLimitError,
    TaskNotFoundError,
    get_research_report_view,
    get_research_status,
    list_research_tasks,
    resume_research,
    start_research,
)
from tragent.worker import enqueue_research_run

router = APIRouter(
    prefix="/api/research",
    tags=["research"],
)
# Re-read the checkpoint after this much silence, so an expired run marker ends the stream.
_EVENT_WAIT_SECONDS = 15.0


class StartResearchRequest(BaseModel):
    """Validated JSON body for creating a new durable research task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Natural-language objective that the Goal Parser converts into a GoalDraft.
    request: str = Field(min_length=1, max_length=2_000)


class ResumeResearchRequest(BaseModel):
    """Validated JSON body used to resume an interrupted or recoverable task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Explicit HITL decision; only statuses that permit the action will accept it.
    action: Literal["approve", "reject"] | None = None
    # Natural-language clarification or requested goal edit supplied by the user.
    answer: ClarificationAnswer | None = None

    @model_validator(mode="after")
    def action_and_answer_are_mutually_exclusive(self) -> Self:
        if self.action is not None and self.answer is not None:
            raise ValueError("a resume request cannot include both an action and an answer")
        return self


@router.get("")
def list_tasks(
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> tuple[ResearchTaskSummary, ...]:
    try:
        return list_research_tasks(user_id=user_id, pool=pool, limit=limit, offset=offset)
    except ResearchApplicationError as exc:
        raise _http_error(exc) from None


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def start_task(
    body: StartResearchRequest,
    user_id: CurrentUserDependency,
    settings: SettingsDependency,
    pool: DatabasePoolDependency,
    redis: RedisDependency,
) -> ResearchTaskView:
    try:
        return start_research(
            body.request,
            user_id=user_id,
            settings=settings,
            pool=pool,
            redis=redis,
            enqueue=enqueue_research_run,
        )
    except ResearchApplicationError as exc:
        raise _http_error(exc) from None


@router.get("/{thread_id}")
def get_task(
    thread_id: UUID4,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    redis: RedisDependency,
) -> ResearchTaskView:
    try:
        return get_research_status(str(thread_id), user_id=user_id, pool=pool, redis=redis)
    except ResearchApplicationError as exc:
        raise _http_error(exc) from None


@router.post("/{thread_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume_task(
    thread_id: UUID4,
    body: ResumeResearchRequest,
    user_id: CurrentUserDependency,
    settings: SettingsDependency,
    pool: DatabasePoolDependency,
    redis: RedisDependency,
) -> ResearchTaskView:
    try:
        return resume_research(
            str(thread_id),
            user_id=user_id,
            settings=settings,
            pool=pool,
            redis=redis,
            enqueue=enqueue_research_run,
            action=body.action,
            answer=body.answer,
        )
    except ResearchApplicationError as exc:
        raise _http_error(exc) from None


@router.get("/{thread_id}/events", response_class=EventSourceResponse)
async def task_events(
    thread_id: UUID4,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    redis: RedisDependency,
    async_redis: AsyncRedisDependency,
) -> AsyncIterator[ServerSentEvent]:
    """Stream live Agent events and task views while a worker has the task queued or running."""

    task_id = str(thread_id)

    def read_view() -> ResearchTaskView:
        return get_research_status(task_id, user_id=user_id, pool=pool, redis=redis)

    async with async_redis.pubsub() as pubsub:
        # Confirm the subscription before reading the checkpoint, so no event between is lost.
        await pubsub.subscribe(task_runs.events_channel(task_id))
        await pubsub.get_message(timeout=_EVENT_WAIT_SECONDS)
        try:
            view = await run_in_threadpool(read_view)
        except ResearchApplicationError as exc:
            yield ServerSentEvent(event="error", data={"detail": str(exc)})
            return
        yield ServerSentEvent(event="result", data=view)
        while view.run_state is not None:
            message = await pubsub.get_message(timeout=_EVENT_WAIT_SECONDS)
            if message is not None:
                event, data = message["data"].split("\n", 1)
                if event == "agent":
                    yield ServerSentEvent(event="agent", raw_data=data)
                    continue
            view = await run_in_threadpool(read_view)
            yield ServerSentEvent(event="result", data=view)


@router.get("/{thread_id}/report")
def get_report(
    thread_id: UUID4,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    response: Response,
) -> ResearchReportView:
    try:
        report = get_research_report_view(str(thread_id), user_id=user_id, pool=pool)
    except ResearchApplicationError as exc:
        raise _http_error(exc) from None
    response.headers["Cache-Control"] = "no-store"
    return report


def _http_error(error: ResearchApplicationError) -> HTTPException:
    detail = str(error)
    if isinstance(error, (TaskNotFoundError, ReportNotFoundError)):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, TaskBusyError):
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(error, TaskLimitError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    else:
        status_code = status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=status_code, detail=detail)


__all__ = ["router"]
