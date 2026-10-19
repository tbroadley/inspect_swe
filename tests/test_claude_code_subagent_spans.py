"""Claude Code sub-agent span reconstruction in LiveConsumer.

Recent Claude Code runs Agent/Task sub-agents in the background by default
(`run_in_background`). The parent's `tool_result` is then an immediate
`async_launched` acknowledgement while the sub-agent keeps making model
calls; completion arrives later as a `task_notification` system event.
"""

from typing import Any

from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
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


def _consumer(monkeypatch: Any) -> tuple[LiveConsumer, _TranscriptStub]:
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


def _launch(consumer: LiveConsumer, background: bool) -> None:
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content="main task")],
            [
                ToolCall(
                    id="toolu_1",
                    function="Agent",
                    arguments={
                        "prompt": PROMPT,
                        "subagent_type": "Explore",
                        "run_in_background": background,
                    },
                )
            ],
        ),
    )


def _subagent_call(consumer: LiveConsumer) -> ModelEvent:
    return _call(consumer, _model_event([ChatMessageUser(content=PROMPT)]))


def _tool_result_line(text: str, tool_use_result: Any) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": text}
            ],
        },
        "tool_use_result": tool_use_result,
    }


ASYNC_ACK = _tool_result_line(
    "Async agent launched successfully.\nagentId: a1b2c3",
    {
        "status": "async_launched",
        "agentId": "a1b2c3",
        "description": "Find TODOs",
        "prompt": PROMPT,
        "outputFile": "/tmp/a1b2c3.output",
    },
)

TASK_NOTIFICATION = {
    "type": "system",
    "subtype": "task_notification",
    "task_id": "a1b2c3",
    "tool_use_id": "toolu_1",
    "status": "completed",
    "output_file": "/tmp/a1b2c3.output",
    "summary": "Found 3 TODOs",
}


def test_foreground_subagent_span_closes_on_tool_result(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, background=False)
    [begin] = stub.span_begins()
    assert (begin.type, begin.name, begin.parent_id) == ("agent", "Explore", "outer")

    assert _subagent_call(consumer).span_id == begin.id

    consumer.process_jsonl_line(
        _tool_result_line("Found 3 TODOs", {"status": "completed"})
    )
    assert [e.id for e in stub.span_ends()] == [begin.id]
    assert _subagent_call(consumer).span_id == "outer"


def test_background_subagent_stays_in_own_span_after_launch_ack(
    monkeypatch: Any,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, background=True)
    [begin] = stub.span_begins()

    consumer.process_jsonl_line(ASYNC_ACK)
    assert stub.span_ends() == []

    # sub-agent keeps working after the parent got its acknowledgement
    assert _subagent_call(consumer).span_id == begin.id
    # the parent's own calls still land on the outer span
    parent = _call(consumer, _model_event([ChatMessageUser(content="main task")]))
    assert parent.span_id == "outer"

    consumer.process_jsonl_line(TASK_NOTIFICATION)
    assert [e.id for e in stub.span_ends()] == [begin.id]
    assert _subagent_call(consumer).span_id == "outer"


def test_background_ack_detected_from_text_without_structured_result(
    monkeypatch: Any,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, background=True)
    line = dict(ASYNC_ACK)
    del line["tool_use_result"]
    consumer.process_jsonl_line(line)
    assert stub.span_ends() == []


def test_reset_closes_background_subagent_still_running(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, background=True)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.reset()
    [begin] = stub.span_begins()
    assert [e.id for e in stub.span_ends()] == [begin.id]
