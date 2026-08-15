"""Authenticated HTTP presentation for the existing Research Agent commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from typing import Annotated, Literal, Self

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from trading.agent.events import AgentEvent
from trading.api import auth
from trading.api.dependencies import DatabasePoolDependency, SettingsDependency
from trading.reporting import ResearchReportView
from trading.research import (
    ResearchStreamItem,
    ResearchTaskStarted,
    ResearchTaskSummary,
    ResearchTaskView,
    get_research_report_view,
    get_research_status,
    list_research_tasks,
    stream_resume_research,
    stream_start_research,
)

router = APIRouter(
    prefix="/api/research",
    tags=["research"],
)


class StartResearchRequest(BaseModel):
    """Validated JSON body for creating a new durable research task."""

    model_config = ConfigDict(extra="forbid")

    # Natural-language objective that the Goal Parser converts into a GoalDraft.
    request: str = Field(min_length=1, max_length=2_000)


class ResumeResearchRequest(BaseModel):
    """Validated JSON body used to resume an interrupted or recoverable task."""

    model_config = ConfigDict(extra="forbid")

    # Explicit HITL decision; only statuses that permit the action will accept it.
    action: Literal["approve", "reject"] | None = None
    # Natural-language clarification or requested goal edit supplied by the user.
    answer: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def action_and_answer_are_mutually_exclusive(self) -> Self:
        if self.action is not None and self.answer is not None:
            raise ValueError("action and answer are mutually exclusive")
        return self


@router.get("")
def list_tasks(
    user_id: Annotated[int, Depends(auth.get_current_user)],
    pool: DatabasePoolDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> tuple[ResearchTaskSummary, ...]:
    try:
        return list_research_tasks(user_id=user_id, pool=pool, limit=limit, offset=offset)
    except ValueError as exc:
        raise _http_error(exc) from None


@router.post("", response_class=EventSourceResponse)
def start_task(
    body: StartResearchRequest,
    user_id: Annotated[int, Depends(auth.get_current_user)],
    settings: SettingsDependency,
    pool: DatabasePoolDependency,
) -> Iterator[ServerSentEvent]:
    yield from _event_stream(
        stream_start_research(body.request, user_id=user_id, settings=settings, pool=pool)
    )


@router.get("/{thread_id}")
def get_task(
    thread_id: UUID4,
    user_id: Annotated[int, Depends(auth.get_current_user)],
    pool: DatabasePoolDependency,
) -> ResearchTaskView:
    try:
        return get_research_status(str(thread_id), user_id=user_id, pool=pool)
    except ValueError as exc:
        raise _http_error(exc) from None


@router.post("/{thread_id}/resume", response_class=EventSourceResponse)
def resume_task(
    thread_id: UUID4,
    body: ResumeResearchRequest,
    user_id: Annotated[int, Depends(auth.get_current_user)],
    settings: SettingsDependency,
    pool: DatabasePoolDependency,
) -> Iterator[ServerSentEvent]:
    yield from _event_stream(
        stream_resume_research(
            str(thread_id),
            user_id=user_id,
            settings=settings,
            pool=pool,
            action=body.action,
            answer=body.answer,
        )
    )


def _event_stream(items: Iterator[ResearchStreamItem]) -> Iterator[ServerSentEvent]:
    with closing(items):
        try:
            for item in items:
                yield _server_sent_event(item)
        except Exception as exc:
            yield ServerSentEvent(event="error", data={"detail": str(exc)})


@router.get("/{thread_id}/report")
def get_report(
    thread_id: UUID4,
    user_id: Annotated[int, Depends(auth.get_current_user)],
    pool: DatabasePoolDependency,
    response: Response,
) -> ResearchReportView:
    try:
        report = get_research_report_view(str(thread_id), user_id=user_id, pool=pool)
    except ValueError as exc:
        raise _http_error(exc) from None
    response.headers["Cache-Control"] = "no-store"
    return report


def _server_sent_event(item: ResearchStreamItem) -> ServerSentEvent:
    # The values "thread", "agent", and "result" are project-defined SSE event names.
    if isinstance(item, ResearchTaskStarted):
        return ServerSentEvent(event="thread", data=item)
    if isinstance(item, AgentEvent):
        return ServerSentEvent(event="agent", data=item)
    return ServerSentEvent(event="result", data=item)


def _http_error(error: ValueError) -> HTTPException:
    detail = str(error)
    status_code = (
        status.HTTP_404_NOT_FOUND
        if detail in {"research task was not found", "research report file was not found"}
        else status.HTTP_400_BAD_REQUEST
    )
    return HTTPException(status_code=status_code, detail=detail)


__all__ = ["router"]
