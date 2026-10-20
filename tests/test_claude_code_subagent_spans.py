"""Claude Code sub-agent span reconstruction in LiveConsumer.

Recent Claude Code runs Agent/Task sub-agents in the background by default
(`run_in_background`). The parent's `tool_result` is then an immediate
`async_launched` acknowledgement while the sub-agent keeps making model
calls; completion arrives later as a `task_notification` system event.
"""

from typing import Any

import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import LiveConsumer

PROMPT = "Find every TODO comment in the repository and summarise them."


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass

    def span_begins(self) -> list[SpanBeginEvent]:
        return [e for e in self.events if isinstance(e, SpanBeginEvent)]

    def span_ends(self) -> list[SpanEndEvent]:
        return [e for e in self.events if isinstance(e, SpanEndEvent)]


def _consumer(monkeypatch: pytest.MonkeyPatch) -> tuple[LiveConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    return LiveConsumer(), stub


def _model_event(
    input: list[ChatMessage], tool_calls: list[ToolCall] | None = None
) -> ModelEvent:
    return ModelEvent(
        model="anthropic/claude-sonnet-4-5",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(
            ChatMessageAssistant(content="ok", tool_calls=tool_calls)
        ),
    )


def _call(consumer: LiveConsumer, event: ModelEvent) -> ModelEvent:
    consumer.on_pending(event)
    consumer.on_complete(event)
    return event


PROMPT_2 = "Run the test suite and report which tests fail and why."
MAIN_TASK = "main task"


# Background vs foreground is decided by the tool_result Claude Code returns,
# not by the `run_in_background` argument, so launches don't set it.
def _launch(
    consumer: LiveConsumer,
    prompts: dict[str, str] | None = None,
    function: str = "Agent",
    main_task: str = MAIN_TASK,
    subagent_type: str = "Explore",
) -> None:
    prompts = prompts or {"toolu_1": PROMPT}
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=main_task)],
            [
                ToolCall(
                    id=tool_use_id,
                    function=function,
                    arguments={"prompt": prompt, "subagent_type": subagent_type},
                )
                for tool_use_id, prompt in prompts.items()
            ],
        ),
    )


def _subagent_call(consumer: LiveConsumer, prompt: str = PROMPT) -> ModelEvent:
    return _call(consumer, _model_event([ChatMessageUser(content=prompt)]))


def _result_block(text: str, tool_use_id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


def _user_line(blocks: list[dict[str, Any]], tool_use_result: Any) -> dict[str, Any]:
    line: dict[str, Any] = {
        "type": "user",
        "message": {"role": "user", "content": blocks},
    }
    if tool_use_result is not None:
        line["tool_use_result"] = tool_use_result
    return line


def _tool_result_line(
    text: str, tool_use_result: Any, tool_use_id: str = "toolu_1"
) -> dict[str, Any]:
    return _user_line([_result_block(text, tool_use_id)], tool_use_result)


def _async_ack(
    tool_use_id: str = "toolu_1",
    prompt: str = PROMPT,
    agent_id: str = "a1b2c3",
    status: str = "async_launched",
) -> dict[str, Any]:
    return _tool_result_line(
        f"Async agent launched successfully.\nagentId: {agent_id}",
        {
            "status": status,
            "agentId": agent_id,
            "description": "Find TODOs",
            "prompt": prompt,
            "outputFile": f"/tmp/{agent_id}.output",
        },
        tool_use_id,
    )


def _notification(
    tool_use_id: str | None = "toolu_1",
    task_id: str = "a1b2c3",
    status: str = "completed",
) -> dict[str, Any]:
    line: dict[str, Any] = {
        "type": "system",
        "subtype": "task_notification",
        "task_id": task_id,
        "status": status,
        "output_file": f"/tmp/{task_id}.output",
        "summary": "done",
    }
    if tool_use_id is not None:
        line["tool_use_id"] = tool_use_id
    return line


ASYNC_ACK = _async_ack()
TASK_NOTIFICATION = _notification()


def test_foreground_subagent_span_closes_on_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    [begin] = stub.span_begins()
    assert (begin.type, begin.name, begin.parent_id) == ("agent", "Explore", "outer")

    assert _subagent_call(consumer).span_id == begin.id

    consumer.process_jsonl_line(
        _tool_result_line("Found 3 TODOs", {"status": "completed", "prompt": PROMPT})
    )
    assert [e.id for e in stub.span_ends()] == [begin.id]
    assert _subagent_call(consumer).span_id == "outer"


def test_background_subagent_stays_in_own_span_after_launch_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    [begin] = stub.span_begins()

    consumer.process_jsonl_line(ASYNC_ACK)
    assert stub.span_ends() == []

    # sub-agent keeps working after the parent got its acknowledgement
    assert _subagent_call(consumer).span_id == begin.id
    # the parent's own calls still land on the outer span
    parent = _call(consumer, _model_event([ChatMessageUser(content=MAIN_TASK)]))
    assert parent.span_id == "outer"

    consumer.process_jsonl_line(TASK_NOTIFICATION)
    assert [e.id for e in stub.span_ends()] == [begin.id]
    assert _subagent_call(consumer).span_id == "outer"


def test_background_ack_detected_from_text_without_structured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    line = dict(ASYNC_ACK)
    del line["tool_use_result"]
    consumer.process_jsonl_line(line)
    assert stub.span_ends() == []


def test_reset_closes_background_subagent_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.reset()
    [begin] = stub.span_begins()
    assert [e.id for e in stub.span_ends()] == [begin.id]


def _ends(stub: _TranscriptStub) -> list[str]:
    return [e.id for e in stub.span_ends()]


def test_task_tool_name_opens_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, function="Task")
    consumer.process_jsonl_line(ASYNC_ACK)
    assert _subagent_call(consumer).span_id == "agent-toolu_1"


@pytest.mark.parametrize("status", ["completed", "failed", "stopped"])
def test_any_notification_status_closes(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_notification(status=status))
    assert _ends(stub) == ["agent-toolu_1"]


def test_notification_before_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(TASK_NOTIFICATION)
    consumer.process_jsonl_line(ASYNC_ACK)
    assert _ends(stub) == ["agent-toolu_1"]
    assert _subagent_call(consumer).span_id == "outer"


def test_unknown_notification_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_notification(tool_use_id="toolu_x", task_id="zzz"))
    consumer.process_jsonl_line(_notification(tool_use_id=None, task_id="zzz"))
    assert _ends(stub) == []


def test_notification_without_tool_use_id_closes_by_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_notification(tool_use_id=None))
    assert _ends(stub) == ["agent-toolu_1"]


def test_concurrent_background_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, {"toolu_1": PROMPT, "toolu_2": PROMPT_2})
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_async_ack("toolu_2", PROMPT_2, "d4e5f6"))
    assert _subagent_call(consumer, PROMPT).span_id == "agent-toolu_1"
    assert _subagent_call(consumer, PROMPT_2).span_id == "agent-toolu_2"

    consumer.process_jsonl_line(_notification("toolu_2", "d4e5f6"))
    assert _ends(stub) == ["agent-toolu_2"]
    assert _subagent_call(consumer, PROMPT).span_id == "agent-toolu_1"
    assert _subagent_call(consumer, PROMPT_2).span_id == "outer"


def test_remote_launched_closes_at_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(status="remote_launched"))
    assert _ends(stub) == ["agent-toolu_1"]


def test_multi_block_line_ignores_line_level_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, {"toolu_1": PROMPT, "toolu_2": PROMPT_2})
    # toolu_1 is acknowledged as background, toolu_2 finished in the
    # foreground; the line-level result describes only one of them
    consumer.process_jsonl_line(
        _user_line(
            [
                _result_block("Async agent launched successfully.", "toolu_1"),
                _result_block("All tests pass.", "toolu_2"),
            ],
            {"status": "async_launched", "agentId": "a1b2c3", "prompt": PROMPT},
        )
    )
    assert _ends(stub) == ["agent-toolu_2"]


def test_parent_quoting_prompt_stays_on_outer_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    main_task = f"Please do this. {PROMPT} Then fix them."
    _launch(consumer, main_task=main_task)
    consumer.process_jsonl_line(ASYNC_ACK)
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
    )
    parent = _call(consumer, _model_event([ChatMessageUser(content=main_task), launch]))
    assert parent.span_id == "outer"
    assert _subagent_call(consumer).span_id == "agent-toolu_1"


def test_foreground_error_result_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(
        _tool_result_line("Error: agent failed", "Error: agent failed")
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_foreground_report_mentioning_launch_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(
        _tool_result_line("Done. Async agent launched for the docs earlier.", None)
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_verbatim_delegation_attributes_to_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, main_task=PROMPT)
    assert _subagent_call(consumer).span_id == "agent-toolu_1"


# Shapes below are from real Claude Code 2.1.303 runs (claude-haiku-4-5).
AGENT_ID = "a6b8c22231413918c"


def _resume_call(
    consumer: LiveConsumer, send_id: str, to: str = AGENT_ID
) -> ModelEvent:
    return _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [
                ToolCall(
                    id=send_id,
                    function="SendMessage",
                    arguments={"to": to, "message": "Now count /etc/group."},
                )
            ],
        ),
    )


def _resumed_subagent_call(consumer: LiveConsumer) -> ModelEvent:
    return _call(
        consumer,
        _model_event(
            [
                ChatMessageUser(content=PROMPT),
                ChatMessageAssistant(content="21 lines."),
                ChatMessageUser(
                    content="The coordinator sent a message while you were "
                    "working:\nNow count /etc/group."
                ),
            ]
        ),
    )


def _finished_background_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveConsumer, _TranscriptStub]:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id=AGENT_ID))
    consumer.process_jsonl_line(_notification(task_id=AGENT_ID))
    return consumer, stub


def test_send_message_resume_opens_new_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _finished_background_agent(monkeypatch)
    _resume_call(consumer, "toolu_send")
    begin = stub.span_begins()[-1]
    assert (begin.id, begin.name, begin.parent_id) == (
        "agent-toolu_1-run2",
        "Explore",
        "outer",
    )
    assert begin.metadata == {"resumes": "agent-toolu_1"}

    # the SendMessage result ("Resuming agent …") does not close it
    consumer.process_jsonl_line(
        _tool_result_line(
            '{"success":true,"message":"Resuming agent a6b8c22"}',
            {"success": True, "resumedAgentId": AGENT_ID},
            "toolu_send",
        )
    )
    assert _resumed_subagent_call(consumer).span_id == "agent-toolu_1-run2"
    # the duplicate task_started for this run does not open another span
    consumer.process_jsonl_line(
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": AGENT_ID,
            "tool_use_id": "toolu_send",
        }
    )
    assert len(stub.span_begins()) == 2

    consumer.process_jsonl_line(_notification("toolu_send", AGENT_ID))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_send_message_by_name_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [
                ToolCall(
                    id="toolu_1",
                    function="Agent",
                    arguments={"prompt": PROMPT, "name": "counter"},
                )
            ],
        ),
    )
    consumer.process_jsonl_line(_async_ack(agent_id=AGENT_ID))
    consumer.process_jsonl_line(_notification(task_id=AGENT_ID))
    _resume_call(consumer, "toolu_send", to="counter")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run2"


def test_send_message_to_running_agent_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id=AGENT_ID))
    _resume_call(consumer, "toolu_send")
    assert len(stub.span_begins()) == 1


def test_task_started_reopens_resumed_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # a message queued while the agent ran makes Claude Code resume it after
    # its notification, under the original tool_use_id
    consumer, stub = _finished_background_agent(monkeypatch)
    consumer.process_jsonl_line(
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": AGENT_ID,
            "tool_use_id": "toolu_1",
        }
    )
    assert stub.span_begins()[-1].id == "agent-toolu_1-run2"
    assert _resumed_subagent_call(consumer).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_notification("toolu_1", AGENT_ID))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_fork_attributed_by_fork_started_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, subagent_type="fork")
    consumer.process_jsonl_line(ASYNC_ACK)
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
    )
    fork = _call(
        consumer,
        _model_event(
            [
                ChatMessageUser(content=MAIN_TASK),
                launch,
                ChatMessageTool(
                    content="Fork started — processing in background",
                    tool_call_id="toolu_1",
                ),
                ChatMessageUser(
                    content="<fork-boilerplate>…</fork-boilerplate>\n\n"
                    f"Your directive: {PROMPT}"
                ),
            ]
        ),
    )
    assert fork.span_id == "agent-toolu_1"
    parent = _call(
        consumer,
        _model_event(
            [
                ChatMessageUser(content=MAIN_TASK),
                launch,
                ChatMessageTool(
                    content="Async agent launched successfully.",
                    tool_call_id="toolu_1",
                ),
            ]
        ),
    )
    assert parent.span_id == "outer"


# ---------------------------------------------------------------------------
# forks launched together, resume edge cases (independent review findings)
# ---------------------------------------------------------------------------


def _launch_many(
    consumer: LiveConsumer, agents: dict[str, tuple[str, str]], extra: bool = False
) -> None:
    calls = [
        ToolCall(id=i, function="Agent", arguments={"prompt": p, "subagent_type": t})
        for i, (p, t) in agents.items()
    ]
    if extra:
        calls.append(ToolCall(id="toolu_bash", function="Bash", arguments={}))
    _call(consumer, _model_event([ChatMessageUser(content=MAIN_TASK)], calls))


def _fork_input(ids: list[str], directive: str) -> list[ChatMessage]:
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id=i, function="Agent", arguments={}) for i in ids],
    )
    return [
        ChatMessageUser(content=MAIN_TASK),
        launch,
        *[
            ChatMessageTool(
                content="Fork started — processing in background", tool_call_id=i
            )
            for i in ids
        ],
        ChatMessageUser(content=f"<fork-boilerplate/>\nYour directive: {directive}"),
    ]


def _send(consumer: LiveConsumer, send_id: str, to: str) -> None:
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [ToolCall(id=send_id, function="SendMessage", arguments={"to": to})],
        ),
    )


def _started(task_id: str, tool_use_id: str, run_id: str) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "task_started",
        "task_id": task_id,
        "tool_use_id": tool_use_id,
        "run_id": run_id,
    }


def test_parallel_forks_pick_their_own_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork"), "t2": (PROMPT_2, "fork")})
    for directive, span in ((PROMPT, "agent-t1"), (PROMPT_2, "agent-t2")):
        call = _call(consumer, _model_event(_fork_input(["t1", "t2"], directive)))
        assert call.span_id == span


def test_fork_beside_regular_agent_and_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_many(
        consumer, {"t1": (PROMPT, "fork"), "t2": (PROMPT_2, "Explore")}, extra=True
    )
    call = _call(
        consumer, _model_event(_fork_input(["t1", "t2", "toolu_bash"], PROMPT))
    )
    assert call.span_id == "agent-t1"


def test_nested_fork_goes_to_innermost(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork")})
    outer_fork = _fork_input(["t1"], PROMPT)
    _call(
        consumer,
        _model_event(
            outer_fork,
            [
                ToolCall(
                    id="t2",
                    function="Agent",
                    arguments={"prompt": PROMPT_2, "subagent_type": "fork"},
                )
            ],
        ),
    )
    assert stub.span_begins()[-1].parent_id == "agent-t1"
    inner = outer_fork + _fork_input(["t2"], PROMPT_2)[1:]
    assert _call(consumer, _model_event(inner)).span_id == "agent-t2"


def test_resumed_fork_uses_new_run(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork")})
    consumer.process_jsonl_line(_async_ack("t1", agent_id="aid"))
    consumer.process_jsonl_line(_notification("t1", "aid"))
    _send(consumer, "s1", "aid")
    history = _fork_input(["t1"], PROMPT) + [
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-t1-run2"


def test_failed_send_message_closes_its_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    _send(consumer, "s1", "aid")
    line = _tool_result_line(
        "<tool_use_error>Error: no agent</tool_use_error>", None, "s1"
    )
    line["message"]["content"][0]["is_error"] = True
    consumer.process_jsonl_line(line)
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]
    # a later, valid resume still gets a span
    _send(consumer, "s2", "aid")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run3"


def test_resumed_run_closes_on_task_id_only_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(None, "aid"))
    _send(consumer, "s1", "aid")
    consumer.process_jsonl_line(_notification(None, "aid"))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_foreground_agent_resumable_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(
        _tool_result_line(
            "21 lines\nagentId: aid", {"status": "completed", "prompt": PROMPT}
        )
    )
    _send(consumer, "s1", "aid")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run2"


def test_resumed_call_before_notification_rolls_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # seen in a real run: the resumed run's first call reaches the bridge
    # before the previous run's notification reaches the stdout reader
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"

    # the late notification for run 1 must not close run 2
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert _ends(stub) == ["agent-toolu_1"]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_lagging_send_message_keys_rolled_over_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")  # run 1's notification not read yet
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_notification("s1", "aid"))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_reset_closes_resumed_run_and_keeps_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    _send(consumer, "s1", "aid")
    consumer.reset()
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]
    # a retry resumes the session, which can still message the agent
    _send(consumer, "s2", "aid")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run3"


def test_same_name_prefers_callers_own_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [
                ToolCall(
                    id="t1", function="Agent", arguments={"prompt": PROMPT, "name": "w"}
                )
            ],
        ),
    )
    # sub-agent t1 launches its own "w"
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=PROMPT)],
            [
                ToolCall(
                    id="t2",
                    function="Agent",
                    arguments={"prompt": PROMPT_2, "name": "w"},
                )
            ],
        ),
    )
    consumer.process_jsonl_line(_tool_result_line("done", None, "t2"))
    consumer.process_jsonl_line(_tool_result_line("done", None, "t1"))
    _send(consumer, "s1", "w")  # from the main agent
    assert stub.span_begins()[-1].id == "agent-t1-run2"


def test_queued_message_resume_after_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # real 2.1.303 trace: SendMessage while the agent runs (queued), run 1's
    # notification closes it, then Claude Code resumes it under the launch id
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21"),
        ChatMessageUser(content="The coordinator sent a message: count /etc/group"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert len(stub.span_begins()) == 2
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def _launch_one(
    consumer: LiveConsumer, tool_use_id: str, prompt: str, name: str | None = None
) -> None:
    args: dict[str, Any] = {"prompt": prompt, "subagent_type": "Explore"}
    if name:
        args["name"] = name
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [ToolCall(id=tool_use_id, function="Agent", arguments=args)],
        ),
    )


def test_relaunch_with_same_prompt_goes_to_new_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_one(consumer, "t1", PROMPT)
    consumer.process_jsonl_line(_async_ack("t1", agent_id="a1"))
    consumer.process_jsonl_line(_notification("t1", "a1"))
    _launch_one(consumer, "t2", PROMPT)
    assert _subagent_call(consumer).span_id == "agent-t2"


def test_longer_prompt_wins_over_the_prompt_it_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    short = "Run the test suite and report."
    _launch_one(consumer, "t1", short)
    _launch_one(consumer, "t2", short + " Then fix the first failure.")
    longer = _subagent_call(consumer, short + " Then fix the first failure.")
    assert longer.span_id == "agent-t2"
    assert _subagent_call(consumer, short).span_id == "agent-t1"


def test_finished_agent_not_matched_without_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e.g. the main agent after compaction, its summary quoting a prompt
    consumer, stub = _consumer(monkeypatch)
    _launch_one(consumer, "t1", PROMPT)
    consumer.process_jsonl_line(_tool_result_line("done", None, "t1"))
    assert _subagent_call(consumer).span_id == "outer"
    assert len(stub.span_begins()) == 1


def test_name_scoped_to_launching_agent_across_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_one(consumer, "t1", PROMPT, name="w")
    # sub-agent t1 launches its own "w" from its span
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=PROMPT)],
            [
                ToolCall(
                    id="t2",
                    function="Agent",
                    arguments={"prompt": PROMPT_2, "name": "w"},
                )
            ],
        ),
    )
    assert stub.span_begins()[-1].parent_id == "agent-t1"
    consumer.process_jsonl_line(_tool_result_line("done", None, "t2"))
    consumer.process_jsonl_line(_tool_result_line("done", None, "t1"))
    # main agent's checkpoint span rotated since t1 was launched
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer-2")
    _send(consumer, "s1", "w")
    assert stub.span_begins()[-1].id == "agent-t1-run2"


def test_task_started_for_new_run_rolls_over(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert _ends(stub) == ["agent-toolu_1"]
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    assert _ends(stub) == ["agent-toolu_1"]
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


@pytest.mark.parametrize(
    "injected",
    [
        # verbatim openings from Claude Code 2.1.303
        "Output token limit hit. Resume directly — no apology, no recap.",
        "Output token limit hit while you were still reasoning, before this "
        "response produced any text or tool call.",
        "Your response was cut off because it exceeded the output token limit. "
        "Continue from where you left off.",
        "[Your previous response reached the output limit with no visible "
        "output. Please continue and produce a brief user-visible response.]",
        "Your response above was cut off mid-stream. Resume directly.",
        "[Your previous response had no visible output. Please continue and "
        "produce a user-visible response.]",
        "Your response above was stopped by a safety classifier — this is not "
        "a tool or API error.",
        "The previous response failed to produce a valid tool call. Please "
        "retry the tool call now.",
        "Goal check-in: «finish the report»",
        "Your task is to create a detailed summary of the conversation so far, "
        "paying close attention to the user's explicit requests.",
        "[structured-output-enforce] call the StructuredOutput tool",
        "You ended the turn without calling SendUserMessage.",
        "Stop hook feedback:\nkeep going",
        "TaskCompleted hook feedback:\nnot done",
        'SubagentStop hook blocking error from command: "check.sh": not done',
        'Stop hook blocking error from command "check.sh": out',
    ],
)
def test_in_run_user_turn_is_not_a_new_run(
    monkeypatch: pytest.MonkeyPatch, injected: str
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="partial"),
        ChatMessageUser(content=injected),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1"
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    assert _ends(stub) == ["agent-toolu_1"]
    assert len(stub.span_begins()) == 1


def test_resume_quoting_an_in_run_message_is_still_a_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # only the start of the turn counts, as in Claude Code
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(
            content="The coordinator sent a message: the log says "
            "'Output token limit hit', please retry"
        ),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1-run2"


def test_in_run_turn_after_system_reminder_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="partial"),
        ChatMessageUser(
            content=[
                ContentText(text="<system-reminder>\nnote\n</system-reminder>"),
                ContentText(text="Output token limit hit. Resume directly."),
            ]
        ),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1"
    assert len(stub.span_begins()) == 1
