"""Authenticated health checks for the Web application."""

from fastapi import APIRouter, Depends

from tragent.api import auth
from tragent.api.dependencies import DatabasePoolDependency, SettingsDependency
from tragent.doctor import CheckResult, run_doctor

router = APIRouter(
    prefix="/api/doctor",
    tags=["doctor"],
    dependencies=[Depends(auth.get_current_user)],
)


@router.post("")
def doctor(
    settings: SettingsDependency,
    pool: DatabasePoolDependency,
) -> tuple[CheckResult, ...]:
    return run_doctor(settings, pool)


__all__ = ["router"]
