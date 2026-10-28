"""Exercise the shared model construction and goal parser contract."""

import json
from typing import Any

import httpx
import pytest

from tragent.config import Settings
from tragent.research.agent.goal import _invoke_goal_draft, _invoke_goal_update, initial_graph_state
from tragent.research.agent.models import AgentBudget, GoalDraft
from tragent.research.llm import build_llm


def test_model_uses_configured_timeout_and_leaves_retries_to_its_callers(
    settings: Settings,
) -> None:
    model = build_llm(settings.model_copy(update={"llm_request_timeout_seconds": 240.0}))
    assert model.request_timeout == 240.0
    assert model.streaming is True
    assert model.root_client.timeout == 240.0
    assert model.root_client.max_retries == 0


@pytest.mark.parametrize("clarification", [True, False])
def test_goal_tool_contract_uses_chat_completions(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    clarification: bool,
) -> None:
    arguments = (
        {"training_sessions": 2, "test_sessions": 1}
        if clarification
        else {"symbol": "AAPL", "bar_interval": "1m"}
    )
    requests: list[dict[str, Any]] = []

    def send(client: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/chat/completions"
        assert payload["tool_choice"] == "required"
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["parallel_tool_calls"] is False
        assert len(payload["tools"]) == 1
        function = payload["tools"][0]["function"]
        assert function["name"] == "GoalDraft"
        assert function["strict"] is (not clarification)

        body = {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": "GoalDraft",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                }
            ],
        }
        message = body["choices"][0].pop("message")
        body["object"] = "chat.completion.chunk"
        body["choices"][0]["delta"] = message
        body["choices"][0]["delta"]["tool_calls"][0]["index"] = 0
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n",
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    model = build_llm(settings)
    if clarification:
        parsed = _invoke_goal_update(
            GoalDraft(symbol="AAPL", bar_interval="1m"),
            ("training_sessions", "test_sessions"),
            "2,1",
            reference_date="2026-09-18",
            model=model,
        )
    else:
        parsed = _invoke_goal_draft(
            initial_graph_state(
                {
                    "thread_id": "00000000-0000-0000-0000-000000000001",
                    "user_id": 1,
                    "request": "Research AAPL with minute bars.",
                    "reference_date": "2026-09-18",
                    "cost_defaults": {"transaction_cost_bps": 5.0, "slippage_bps": 5.0},
                    "position_tranches_default": 5,
                    "budget": AgentBudget().model_dump(mode="json"),
                },
            ),
            model=model,
        )

    assert len(requests) == 1
    assert isinstance(parsed, GoalDraft)
    assert parsed.model_dump(exclude_unset=True) == arguments
