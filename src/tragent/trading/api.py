"""The signed-in user's IBKR paper account, orders, deployed strategies, and a WebSocket of their
changes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import UUID4, BaseModel, Field

from tragent.api.auth import CurrentUserDependency, user_from_token
from tragent.api.dependencies import (
    DatabasePoolDependency,
    KafkaProducerDependency,
    RedisDependency,
)
from tragent.db import DatabasePool
from tragent.research.service import (
    ResearchApplicationError,
    get_research_report_view,
    get_research_status,
)
from tragent.trading import containers, deployments, history, messaging, orders, store
from tragent.trading.broker import status_key

router = APIRouter(prefix="/api/trading", tags=["trading"])


class BrokerCredentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1)


class DeployRequest(BaseModel):
    thread_id: UUID4
    capital: float = Field(gt=0)


class StopRequest(BaseModel):
    close_position: bool


class DownloadLink(BaseModel):
    url: str


class DeploymentList(BaseModel):
    next_run_at: datetime
    deployments: list[deployments.DeploymentView]


class Connection(BaseModel):
    connected: bool
    error: str | None


class AccountView(BaseModel):
    username: str
    connection: Connection
    balances: store.Balances | None
    positions: list[store.Position]
    synced_at: datetime | None


@router.get("/account")
def get_account(
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    redis: RedisDependency,
) -> AccountView:
    overview = store.account_overview(pool, user_id)
    if overview is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No IBKR account is linked")
    # The broker process renews the status every round, so a missing one means it is not running.
    raw_status = redis.get(status_key(user_id))
    connection = (
        Connection.model_validate(json.loads(raw_status))
        if raw_status
        else Connection(connected=False, error="The broker process is not running")
    )
    snapshot = overview.snapshot
    return AccountView(
        username=overview.username,
        connection=connection,
        balances=snapshot.balances if snapshot else None,
        positions=list(snapshot.positions) if snapshot else [],
        synced_at=overview.synced_at,
    )


@router.put("/account", status_code=status.HTTP_204_NO_CONTENT)
def link_account(
    body: BrokerCredentials,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
) -> None:
    """Log a Gateway in with the paper login, and save the login only once IBKR accepts it."""

    if not containers.log_in(user_id, body.username, body.password):
        # A rejected change of login brings back the Gateway of the login still saved.
        saved = store.account_credentials(pool, user_id)
        if saved is not None:
            containers.run_gateway(user_id, *saved)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "IBKR did not accept this login. Check the paper username and password.",
        )
    store.save_account(pool, user_id, body.username, body.password)


@router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
def unlink_account(user_id: CurrentUserDependency, pool: DatabasePoolDependency) -> None:
    if not store.delete_account(pool, user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No IBKR account is linked")
    containers.stop_gateway(user_id)


@router.post("/orders", status_code=status.HTTP_201_CREATED)
def place_order(
    body: orders.OrderRequest,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    producer: KafkaProducerDependency,
    response: Response,
) -> orders.Order:
    """Save the order and hand it to the broker process; a repeated request returns the same order."""

    try:
        order, created = orders.create_order(pool, user_id, body)
    except orders.NoAccountError:
        raise HTTPException(status.HTTP_409_CONFLICT, "Link an IBKR account first") from None
    if created:
        messaging.send_command(producer, order, "place")
    else:
        response.status_code = status.HTTP_200_OK
    return order


@router.get("/orders")
def list_orders(
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[orders.Order]:
    return orders.list_orders(pool, user_id, limit)


@router.delete("/orders/{order_id}", status_code=status.HTTP_202_ACCEPTED)
def cancel_order(
    order_id: int,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    producer: KafkaProducerDependency,
) -> orders.Order:
    """Ask the broker process to cancel the order; the WebSocket reports when it is cancelled."""

    order = orders.get_order(pool, order_id)
    if order is None or order.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Order not found")
    if order.status in orders.FINAL_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, f"The order is already {order.status}")
    messaging.send_command(producer, order, "cancel")
    return order


@router.post("/deployments", status_code=status.HTTP_201_CREATED)
def deploy_strategy(
    body: DeployRequest,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    redis: RedisDependency,
) -> deployments.Deployment:
    """Deploy the strategy a research task's Critic accepted to the user's paper account."""

    thread_id = str(body.thread_id)
    try:
        report = get_research_report_view(thread_id, user_id=user_id, pool=pool)
        task = get_research_status(thread_id, user_id=user_id, pool=pool, redis=redis)
        return deployments.deploy(
            pool, user_id, thread_id, report, task.goal["position_tranches"], body.capital
        )
    except (ResearchApplicationError, deployments.NotDeployableError) as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None


@router.get("/deployments")
def list_deployments(
    user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> DeploymentList:
    return DeploymentList(
        next_run_at=deployments.next_run(datetime.now(UTC)),
        deployments=deployments.list_deployments(pool, user_id),
    )


@router.get("/deployments/{deployment_id}/runs")
def list_runs(
    deployment_id: int, user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> list[deployments.Run]:
    _own_deployment(pool, deployment_id, user_id)
    return deployments.list_runs(pool, deployment_id)


@router.post("/deployments/{deployment_id}/pause", status_code=status.HTTP_204_NO_CONTENT)
def pause_deployment(
    deployment_id: int, user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> None:
    if _own_deployment(pool, deployment_id, user_id).status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, "Only an active deployment can be paused")
    deployments.set_status(pool, deployment_id, "paused")


@router.post("/deployments/{deployment_id}/resume", status_code=status.HTTP_204_NO_CONTENT)
def resume_deployment(
    deployment_id: int, user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> None:
    if _own_deployment(pool, deployment_id, user_id).status != "paused":
        raise HTTPException(status.HTTP_409_CONFLICT, "Only a paused deployment can be resumed")
    deployments.set_status(pool, deployment_id, "active")


@router.post("/deployments/{deployment_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
def stop_deployment(
    deployment_id: int,
    body: StopRequest,
    user_id: CurrentUserDependency,
    pool: DatabasePoolDependency,
    producer: KafkaProducerDependency,
) -> None:
    """Stop the deployment for good; closing its position sells the shares its fills hold."""

    deployment = _own_deployment(pool, deployment_id, user_id)
    if deployment.status == "stopped":
        raise HTTPException(status.HTTP_409_CONFLICT, "The deployment is already stopped")
    deployments.stop(pool, producer, deployment, close_position=body.close_position)


def _own_deployment(pool: DatabasePool, deployment_id: int, user_id: int) -> deployments.Deployment:
    deployment = deployments.get_deployment(pool, deployment_id)
    if deployment is None or deployment.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Deployment not found")
    return deployment


@router.post("/statements", status_code=status.HTTP_201_CREATED)
def upload_statement(
    file: UploadFile, user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> history.Statement:
    """Import the stock trades of an IBKR Activity Statement CSV, each trade once."""

    data = file.file.read(history.MAX_STATEMENT_BYTES + 1)
    if len(data) > history.MAX_STATEMENT_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "A statement is at most 10 MB")
    try:
        return history.import_statement(pool, user_id, file.filename or "statement.csv", data)
    except history.StatementError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from None


@router.get("/statements")
def list_statements(
    user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> list[history.Statement]:
    return history.list_statements(pool, user_id)


@router.get("/statements/{statement_id}/download")
def download_statement(
    statement_id: int, user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> DownloadLink:
    url = history.statement_download_url(pool, user_id, statement_id)
    if url is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Statement not found")
    return DownloadLink(url=url)


@router.get("/history")
def trade_history(
    user_id: CurrentUserDependency, pool: DatabasePoolDependency
) -> list[history.HistoryTrade]:
    return history.trade_history(pool, user_id)


@router.post("/exports")
def export_history(user_id: CurrentUserDependency, pool: DatabasePoolDependency) -> DownloadLink:
    """A CSV of the trade history, downloaded straight from the object store."""

    return DownloadLink(url=history.export_history(pool, user_id))


@router.websocket("/ws")
async def trading_events(websocket: WebSocket, token: str) -> None:
    """The user's orders as they change and their balances and positions as IBKR updates them.

    Orders come as the REST API returns them, and account snapshots as balances, positions, and
    the time they were read.

    Browsers cannot set headers on a WebSocket, so the access token comes in the query string.
    """

    user_id = user_from_token(token, websocket.app.state.settings)
    if user_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    # Subscribed before accepting, so no event between the two is missed.
    with websocket.app.state.trading_feed.subscribe(user_id, websocket):
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass


__all__ = ["router"]
