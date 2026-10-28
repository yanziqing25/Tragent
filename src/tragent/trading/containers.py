"""One IB Gateway container per user, logged in to their paper account with the saved credentials.

The community image runs IB Gateway headless and logs it in with IBC. Docker keeps each container
running across crashes and restarts, so the platform only creates and removes them.
"""

import time
from functools import cache

import docker
from docker.errors import NotFound
from docker.models.containers import Container

from tragent.config import load_settings

_IMAGE = "ghcr.io/gnzsnz/ib-gateway:latest"
# The image relays the paper API from the container's localhost to this port.
_PAPER_API_PORT = 4004
_FIRST_HOST_PORT = 4100
# IBC logs these lines; IBKR reports a rejected login in a dialog titled just "Gateway".
_LOGGED_IN = "IBC: Login has completed"
_LOGIN_FAILED = "IBC: detected dialog entitled: Gateway; event=Opened"
# A login that IBKR accepts completes in under a minute.
_LOGIN_SECONDS = 120


def gateway_address(user_id: int) -> tuple[str, int]:
    """Where the broker reaches a user's Gateway: by name on the Gateways' network, if they have
    one, else on the localhost port it is published at."""

    if _network():
        return _container_name(user_id), _PAPER_API_PORT
    return "127.0.0.1", _FIRST_HOST_PORT + user_id


def log_in(user_id: int, username: str, password: str) -> bool:
    """Replace the user's Gateway with one for these credentials, kept only if IBKR accepts them."""

    container = run_gateway(user_id, username, password)
    if _wait_for_login(container):
        return True
    container.remove(force=True)
    return False


def run_gateway(user_id: int, username: str, password: str) -> Container:
    """Replace the user's Gateway with one logging in as these credentials."""

    stop_gateway(user_id)
    return docker.from_env().containers.run(
        _IMAGE,
        name=_container_name(user_id),
        detach=True,
        environment={
            "TWS_USERID": username,
            "TWS_PASSWORD": password,
            "TRADING_MODE": "paper",
            "READ_ONLY_API": "no",
            # A daily restart keeps the session alive without a fresh two-factor approval.
            "AUTO_RESTART_TIME": "11:59 PM",
            "TWOFA_TIMEOUT_ACTION": "restart",
        },
        ports={f"{_PAPER_API_PORT}/tcp": ("127.0.0.1", _FIRST_HOST_PORT + user_id)},
        network=_network(),
        restart_policy={"Name": "unless-stopped"},
    )


def stop_gateway(user_id: int) -> None:
    try:
        docker.from_env().containers.get(_container_name(user_id)).remove(force=True)
    except NotFound:
        pass


def _wait_for_login(container: Container) -> bool:
    deadline = time.monotonic() + _LOGIN_SECONDS
    while time.monotonic() < deadline:
        logs = container.logs().decode(errors="replace")
        if _LOGGED_IN in logs:
            return True
        if _LOGIN_FAILED in logs:
            return False
        time.sleep(1)
    return False


@cache
def _network() -> str | None:
    return load_settings().gateway_network


def _container_name(user_id: int) -> str:
    return f"tragent-gateway-{user_id}"


__all__ = ["gateway_address", "log_in", "run_gateway", "stop_gateway"]
