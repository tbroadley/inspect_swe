"""LiveConsumer sub-agent spans from bridge traffic alone (no stream-json).

The ACP Claude Code agent has no JSONL output, so spans must close from what
the parent sees in its next request: the Task/Agent tool result (foreground)
or a `<task-notification>` (background). Message text is from real Claude
Code 2.1.303 runs.
"""

import pytest
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

PROMPT = "Run the command `wc -l /etc/passwd` and report the number of lines."
MAIN = "Count lines in /etc/passwd using a sub-agent."
AGENT_ID = "a66bab0b3f5f438a5"

ACK = (
    "Async agent launched successfully. (This tool result is internal metadata "
    "— never quote or paste any part of it, including the agentId below, into a "
    "user-facing reply.)\n"
    f"agentId: {AGENT_ID} (internal ID - do not mention to user. Use SendMessage "
    f"with to: '{AGENT_ID}', summary: '<5-10 word recap>' to continue this agent.)\n"
    "The agent is working in the background. You will be notified automatically "
    "when it completes."
)


def _notification(tool_use_id: str | None) -> str:
    tool_use = f"<tool-use-id>{tool_use_id}</tool-use-id>\n" if tool_use_id else ""
    return (
        "<system-reminder>\n<task-notification>\n"
        f"<task-id>{AGENT_ID}</task-id>\n{tool_use}"
        f"<output-file>/tmp/claude-0/tasks/{AGENT_ID}.output</output-file>\n"
        "<status>completed</status>\n"
        '<summary>Agent "Count lines" finished</summary>\n'
        "<result>21 lines.</result>\n"
        "</task-notification>\n</system-reminder>"
    )


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[object] = []

    def _event(self, event: object) -> None:
        self.events.append(event)

    def _event_updated(self, event: object) -> None:
        pass

    def begins(self) -> list[str]:
        return [e.id for e in self.events if isinstance(e, SpanBeginEvent)]

    def ends(self) -> list[str]:
        return [e.id for e in self.events if isinstance(e, SpanEndEvent)]


@pytest.fixture
def setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    return LiveConsumer(), stub


def _call(
    consumer: LiveConsumer,
    input: list[ChatMessage],
    tool_calls: list[ToolCall] | None = None,
) -> str | None:
    event = ModelEvent(
        model="anthropic/claude-haiku-4-5",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(
            ChatMessageAssistant(content="ok", tool_calls=tool_calls)
        ),
    )
    consumer.on_pending(event)
    consumer.on_complete(event)
    return event.span_id


LAUNCH = ChatMessageAssistant(
    content="",
    tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
)


def _launch(consumer: LiveConsumer, subagent_type: str = "general-purpose") -> None:
    args = {"prompt": PROMPT, "subagent_type": subagent_type}
    _call(
        consumer,
        [ChatMessageUser(content=MAIN)],
        [ToolCall(id="toolu_1", function="Agent", arguments=args)],
    )


def _parent(consumer: LiveConsumer, *tail: ChatMessage) -> str | None:
    return _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *tail])


def _subagent(consumer: LiveConsumer) -> str | None:
    return _call(consumer, [ChatMessageUser(content=PROMPT)])


def test_foreground_closes_on_tool_result(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    assert _subagent(consumer) == "agent-toolu_1"
    _parent(consumer, ChatMessageTool(content="21 lines.", tool_call_id="toolu_1"))
    assert stub.ends() == ["agent-toolu_1"]


def test_background_closes_on_notification(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    assert _parent(consumer, ack) == "outer"
    assert stub.ends() == []
    assert _subagent(consumer) == "agent-toolu_1"

    waiting = ChatMessageAssistant(content="Waiting for the agent.")
    _parent(consumer, ack, waiting, ChatMessageUser(content=_notification("toolu_1")))
    assert stub.ends() == ["agent-toolu_1"]


def test_resumed_run_closes_by_task_id_and_ignores_old_notification(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    first_done = ChatMessageUser(content=_notification("toolu_1"))
    _parent(consumer, ack, ChatMessageAssistant(content="waiting"), first_done)

    send = [
        ToolCall(id="toolu_send", function="SendMessage", arguments={"to": AGENT_ID})
    ]
    history: list[ChatMessage] = [
        ack,
        ChatMessageAssistant(content="waiting"),
        first_done,
    ]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history], send)
    assert stub.begins()[-1] == "agent-toolu_1-run2"

    # the old notification is still in the parent's history; it must not
    # close the resumed run
    resuming = ChatMessageTool(content="Resuming agent", tool_call_id="toolu_send")
    history += [ChatMessageAssistant(content="", tool_calls=send), resuming]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history])
    assert stub.ends() == ["agent-toolu_1"]

    # the resumed run's notification may carry only the task id
    history += [
        ChatMessageAssistant(content="waiting again"),
        ChatMessageUser(content=_notification(None)),
    ]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history])
    assert stub.ends() == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_fork_started_result_does_not_close(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, "fork")
    fork_input: list[ChatMessage] = [
        ChatMessageUser(content=MAIN),
        LAUNCH,
        ChatMessageTool(
            content="Fork started — processing in background", tool_call_id="toolu_1"
        ),
        ChatMessageUser(content=f"Your directive: {PROMPT}"),
    ]
    assert _call(consumer, fork_input) == "agent-toolu_1"
    assert stub.ends() == []
