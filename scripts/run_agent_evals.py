"""Run the small real-model V2 Research/Critic evaluation in LangSmith."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from datetime import date
from uuid import NAMESPACE_URL, UUID, uuid5

from agent_eval_cases import (
    CASES,
    AgentEvalCase,
    CaseContractError,
    run_case,
    validate_cases,
)
from langsmith import Client

from tragent import redis_client
from tragent.config import ConfigurationError, Settings, load_settings
from tragent.db import DatabasePool, build_connection_pool
from tragent.research.llm import build_llm
from tragent.research.persistence import setup_persistence
from tragent.research.service import ResearchTaskView, resume_research, run_research, start_research

DATASET_NAME = "trading-v2-agent-decisions-v2"
EXPERIMENT_PREFIX = "trading-v2-agent-decisions"
_EXAMPLE_NAMESPACE = uuid5(NAMESPACE_URL, DATASET_NAME)
_SMOKE_USER_ID = 1


class EvalError(RuntimeError):
    """A bounded developer-facing evaluation failure."""


def _example_id(case_id: str) -> UUID:
    return uuid5(_EXAMPLE_NAMESPACE, case_id)


def _sync_dataset(client: Client, cases: Mapping[str, AgentEvalCase]) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=("Eight real-model Research/Critic decision cases for Tragent."),
            metadata={"owner": "tragent", "contract_version": 2},
        )

    existing_ids: set[str] = set()
    for example in client.list_examples(dataset_id=dataset.id):
        inputs = example.inputs if isinstance(example.inputs, Mapping) else {}
        case_id = inputs.get("case_id")
        if not isinstance(case_id, str) or case_id not in cases:
            raise EvalError(
                "The LangSmith Dataset contains an unknown example; no remote data was changed."
            )
        if example.id != _example_id(case_id) or inputs.get("agent") != cases[case_id].agent:
            raise EvalError(
                f"LangSmith example {case_id} conflicts with the checked-in v2 case contract."
            )
        outputs = example.outputs if isinstance(example.outputs, Mapping) else {}
        if (
            dict(inputs) != cases[case_id].inputs()
            or dict(outputs) != cases[case_id].reference_outputs()
        ):
            raise EvalError(
                f"LangSmith example {case_id} differs from the checked-in v2 case contract."
            )
        existing_ids.add(case_id)

    missing = [case for case in CASES if case.case_id not in existing_ids]
    if missing:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "id": _example_id(case.case_id),
                    "inputs": case.inputs(),
                    "outputs": case.reference_outputs(),
                    "metadata": {"source": "checked-in", "contract_version": 2},
                }
                for case in missing
            ],
        )


def _policy_evaluator(
    inputs: dict[str, object],
    outputs: dict[str, object],
    reference_outputs: dict[str, object],
) -> list[dict[str, object]]:
    del inputs
    actions = outputs.get("actions")
    normalized_actions = actions if isinstance(actions, list) else []
    names = [
        action.get("tool_name")
        for action in normalized_actions
        if isinstance(action, Mapping) and isinstance(action.get("tool_name"), str)
    ]
    allowed = set(reference_outputs.get("allowed_tools", []))
    forbidden = set(reference_outputs.get("forbidden_tools", []))
    max_decisions = reference_outputs.get("max_decisions")
    results: list[dict[str, object]] = [
        {"key": "valid_protocol", "score": outputs.get("protocol_valid") is True},
        {
            "key": "tool_policy",
            "score": bool(names)
            and len(names) == len(normalized_actions)
            and all(name in allowed and name not in forbidden for name in names),
        },
        {
            "key": "budget_policy",
            "score": isinstance(max_decisions, int)
            and isinstance(outputs.get("decisions"), int)
            and outputs["decisions"] <= max_decisions
            and isinstance(outputs.get("tool_calls"), int)
            and outputs["tool_calls"] <= max_decisions,
        },
    ]

    allowed_verdicts = reference_outputs.get("allowed_verdicts")
    if isinstance(allowed_verdicts, list) and allowed_verdicts:
        results.append(
            {"key": "verdict_policy", "score": outputs.get("verdict") in allowed_verdicts}
        )
        results.append(
            {
                "key": "evidence_review",
                "score": "inspect_retrieved_context" in names
                and "submit_critic_review" in names
                and names.index("inspect_retrieved_context") < names.index("submit_critic_review"),
            }
        )

    if reference_outputs.get("require_training_backtest") is True:
        first = normalized_actions[0] if normalized_actions else {}
        arguments = first.get("arguments") if isinstance(first, Mapping) else None
        results.append(
            {
                "key": "required_training_backtest",
                "score": first.get("tool_name") == "run_backtest"
                and isinstance(arguments, Mapping)
                and arguments.get("stage") == "training",
            }
        )

    forbidden_parameters = reference_outputs.get("forbid_exact_parameters")
    if isinstance(forbidden_parameters, Mapping):
        repeated = any(
            isinstance(action, Mapping)
            and action.get("tool_name") == "run_backtest"
            and isinstance(action.get("arguments"), Mapping)
            and action["arguments"].get("stage") == "training"
            and action["arguments"].get("parameters") == forbidden_parameters
            for action in normalized_actions
        )
        results.append({"key": "no_exact_repeat", "score": not repeated})
    return results


def _experiment_failed(rows: list[object]) -> bool:
    failed = False
    for row in rows:
        if not isinstance(row, Mapping):
            return True
        example = row.get("example")
        inputs = getattr(example, "inputs", {})
        case_id = inputs.get("case_id", "unknown") if isinstance(inputs, Mapping) else "unknown"
        failed_keys: list[str] = []
        if getattr(row.get("run"), "error", None):
            failed_keys.append("target_error")
        batch = row.get("evaluation_results")
        values = batch.get("results", []) if isinstance(batch, Mapping) else []
        if not values:
            failed_keys.append("missing_scores")
        failed_keys.extend(
            str(getattr(value, "key", "unknown_score"))
            for value in values
            if getattr(value, "score", None) is not True
        )
        if failed_keys:
            failed = True
            print(f"Failed case {case_id}: {', '.join(failed_keys)}")
    return failed


def _run_experiment(settings: Settings, cases: Mapping[str, AgentEvalCase]) -> bool:
    client = Client(api_key=settings.langsmith_api_key.get_secret_value())
    try:
        _sync_dataset(client, cases)
        model = build_llm(settings)
        results = client.evaluate(
            lambda inputs: run_case(inputs, model, cases),
            data=DATASET_NAME,
            evaluators=[_policy_evaluator],
            experiment_prefix=EXPERIMENT_PREFIX,
            description="Deterministic policy checks for the V2 Research and Critic Agents.",
            metadata={"model": settings.llm_model, "case_contract": "v2"},
            max_concurrency=0,
            error_handling="log",
        )
        rows = list(results)
        print(f"LangSmith Agent Eval: {len(rows)}/{len(CASES)} cases completed.")
        if results.url:
            print(f"Experiment: {results.url}")
        return len(rows) == len(CASES) and not _experiment_failed(rows)
    finally:
        client.close()


def _validate_smoke(view: ResearchTaskView) -> None:
    if view.status not in {"selected", "no_strategy", "critic_rejected"}:
        raise EvalError("The full smoke did not reach a Critic-reviewed terminal.")
    if view.critic_round < 1 or view.critic_verdict not in {"accept", "reject"}:
        raise EvalError("The full smoke did not produce a final Critic verdict.")
    if not view.report_exists:
        raise EvalError("The full smoke did not generate its report.")


def _run_smoke(settings: Settings, pool: DatabasePool) -> None:
    request = (
        "研究 AAPL 从 2024-01-01 到 2024-06-30 的日线，在 40% 最大回撤约束下比较 RSI，"
        "并与 SPY 进行比较；手续费 5 bps，滑点 5 bps"
    )
    redis = redis_client.connect(settings)

    def run_inline(thread_id: str, user_id: int, resume: dict[str, str] | None) -> None:
        """Run each queued step in this process instead of a Celery worker."""

        run_research(
            thread_id, user_id=user_id, resume=resume, settings=settings, pool=pool, redis=redis
        )

    started = start_research(
        request,
        user_id=_SMOKE_USER_ID,
        settings=settings,
        pool=pool,
        redis=redis,
        enqueue=run_inline,
        reference_date=date(2024, 6, 30),
    )
    print(f"Full smoke thread: {started.thread_id}")
    if started.status != "awaiting_confirmation":
        raise EvalError("The full smoke did not stop at the expected HITL confirmation.")
    completed = resume_research(
        started.thread_id,
        user_id=_SMOKE_USER_ID,
        pool=pool,
        redis=redis,
        enqueue=run_inline,
        action="approve",
    )
    if completed.status == "running" and completed.can_resume:
        completed = resume_research(
            started.thread_id,
            user_id=_SMOKE_USER_ID,
            pool=pool,
            redis=redis,
            enqueue=run_inline,
        )
    _validate_smoke(completed)
    print(f"Full smoke: {completed.thread_id} -> {completed.status} ({completed.critic_verdict}).")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the eight-case LangSmith V2 Agent evaluation."
    )
    parser.add_argument(
        "--with-smoke",
        action="store_true",
        help="also run one full HITL Research-to-Critic task after the Dataset passes",
    )
    return parser


def main() -> int:
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    args = _parser().parse_args()
    try:
        cases = validate_cases()
        settings = load_settings()
        if not _run_experiment(settings, cases):
            print("Agent Eval failed one or more deterministic policy checks.")
            return 1
        print("Agent Eval passed all deterministic policy checks.")
        if args.with_smoke:
            with build_connection_pool(settings) as pool:
                pool.wait(timeout=settings.postgres_pool_timeout_seconds)
                setup_persistence(pool, settings)
                _run_smoke(settings, pool)
    except ConfigurationError as error:
        print(str(error))
        return 1
    except (CaseContractError, EvalError) as error:
        print(f"Agent Eval stopped safely: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
