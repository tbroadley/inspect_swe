"""Real-time consumer of Claude Code JSONL output.

Two responsibilities, both driven from the same `LiveConsumer` instance:

1. **`ModelEventSink`** — installed on the agent bridge so the bridge hands
   us every `ModelEvent` for routing instead of emitting it to the transcript
   itself. We attribute each event to the correct agent span at `on_pending`
   time, then forward to the transcript with the attributed span_id.

   **Attribution mechanism (substring match against pending sub-agents)**:

   Claude Code 2.1.x makes sub-agents opaque in JSONL — sub-agent model
   calls have no `parent_tool_use_id`, no separate `session_id`, and no
   `isSidechain` marker. So we cannot use JSONL alone to drive span
   open/close for sub-agents.

   When a bridge call's output contains Task/Agent tool_calls (i.e. the
   parent agent is spawning sub-agents), `on_complete` does two things,
   synchronously, *before* the bridge response is sent back to Claude
   Code:

     1. Open an agent `SpanBeginEvent` for each Task/Agent tool_use, with
        `parent_id` = the parent call's span_id and span_id =
        `agent-{tool_use_id}`.
     2. Register the sub-agent in `_pending_subagents` (mapping
        `tool_use_id → prompt`).

   When sub-agent's first bridge call arrives, `_attribute` scans its
   first user message text for any pending sub-agent prompt as a
   substring. A single hit identifies the sub-agent and we look up its
   already-open span. Zero hits → main-agent call (outer span). Multiple
   hits (rare; concurrent sub-agents with substring-overlapping prompts)
   → outer span as defensive default.

   Doing both open + register in `on_complete` (rather than from JSONL)
   eliminates a race: the sub-agent's bridge call arrives at the bridge
   server ~1–2 seconds before our stdout reader processes the parent's
   `assistant` JSONL line, so a JSONL-driven span open would miss every
   first call.

   This works for every sub-agent call (not just the first) because
   Claude Code re-sends the sub-agent's full conversation history on
   each request, with the original Task prompt always at `input[0]`.
   `_pending_subagents` and the open span are cleared in `_handle_user`
   when the matching `tool_result` arrives — except for background agents
   (`run_in_background`, the default in recent Claude Code), whose
   `tool_result` is an immediate `async_launched` acknowledgement while the
   sub-agent keeps running. Those stay open until the `task_notification`
   system event for the same `tool_use_id` (or `reset()`).

   Two kinds of sub-agent need more than prompt matching:

   - **Forks** (`subagent_type: "fork"`) inherit the parent's history, so
     the prompt is not in their first message. Claude Code answers every
     tool_use in the launching message with a "Fork started" placeholder in
     the fork's history, followed by the fork's directive (its prompt).
   - **Resumed runs.** A finished background agent resumes when sent a
     message (SendMessage, or a message queued while it ran). Each run gets
     its own span, `agent-{tool_use_id}-run{N}`. It is opened in
     `on_complete` for a SendMessage, or lazily in `on_pending` when a call
     for the agent shows a run boundary (a text-only assistant turn followed
     by a user turn) that we have no span for yet, with JSONL `task_started`
     as a fallback.

2. **JSONL consumer** — `process_jsonl_line` reads each line printed by
   Claude Code's `--output-format stream-json` and emits agent
   `SpanEndEvent` (on `tool_result` for foreground Task/Agent, on
   `task_notification` for background ones), and emits
   `CompactionEvent` for `compact_boundary` system events. `task_started`
   binds task ids and run ids, and only opens a span for a resumed run that
   the bridge has not shown us yet.
"""

import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, cast

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.model._model_output import StopReason
from inspect_ai.util._span import current_span_id

from .toolview import tool_view

logger = getLogger(__name__)


# Minimum prompt length to consider for substring matching. Short prompts
# could accidentally appear in unrelated content; this guards against false
# positives while still catching every plausible Task prompt (which are
# typically full sentences).
_MIN_PROMPT_LENGTH = 16


@dataclass
class _OpenAgent:
    """An agent span currently open (Task tool_use seen, sub-agent not done)."""

    span_id: str
    # tool_use_id of the Task/Agent call that launched the sub-agent
    origin: str
    # keyed by a SendMessage tool_use_id: that tool's (successful) result
    # means the run started, not that it finished
    via_send_message: bool = False
    # Claude Code's run id (from task_started), and those of earlier runs
    # whose late notifications must not close this one
    run_id: str | None = None
    stale_run_ids: set[str] = field(default_factory=set)


@dataclass
class _Launched:
    """A sub-agent launched this attempt, kept after it finishes.

    A finished background agent can be resumed (SendMessage), and the resumed
    run needs a span of its own. Kept across `reset()`: a retry resumes the
    Claude Code session, which can still message these agents.
    """

    prompt: str
    span_name: str
    description: str
    parent_span_id: str | None
    name: str | None = None
    agent_id: str | None = None
    fork: bool = False
    runs: int = 0
    # SendMessage seen while the previous run still looked open; keys the
    # run it starts if that run opens lazily
    pending_send: str | None = None
    # span of the latest run, for calls that arrive after it closed
    span_id: str | None = None
    # launching tool_use_id of the sub-agent that launched this one (None:
    # the main agent); SendMessage names are scoped to the sender's children
    launcher: str | None = None


class LiveConsumer(ModelEventSink):
    """Sink + JSONL consumer.

    The bridge calls `on_pending` / `on_complete` for every `ModelEvent`.
    The runner loop calls `process_jsonl_line` for every JSONL line printed
    by Claude Code.
    """

    def __init__(self) -> None:
        # tool_use_id → _OpenAgent for currently-open agent spans (Task/Agent
        # tool_use blocks we've SpanBegin'd, not yet SpanEnd'd).
        self._open_agents: dict[str, _OpenAgent] = {}

        # tool_use_id → Task prompt for sub-agents currently RUNNING. Populated
        # in `on_complete` when a parent's output contains Task/Agent tool_calls
        # (synchronously, before the response is sent back to Claude Code, so
        # the entry is ready before any sub-agent can make a bridge call).
        # Cleared when the sub-agent finishes (see `_close_agent`).
        self._pending_subagents: dict[str, str] = {}

        # background agent task_id → tool_use_id, for task_notifications that
        # carry no tool_use_id
        self._task_ids: dict[str, str] = {}

        # launching tool_use_id → _Launched, for every sub-agent this session
        self._launched: dict[str, _Launched] = {}

        # Track which event objects we've already _event()'d so on_complete
        # knows whether to emit _event_updated (yes if we emitted) vs swallow
        # (no if we didn't — shouldn't happen with current logic but defensive).
        self._emitted_events: set[int] = set()

        # Stop reason of the most recent completed ModelEvent. Used by the
        # runner loop to distinguish an Anthropic refusal (content_filter)
        # from a genuine scaffold crash when Claude Code exits non-zero.
        self._last_stop_reason: StopReason | None = None

    @property
    def last_stop_reason(self) -> StopReason | None:
        """Stop reason of the last completed model event this attempt."""
        return self._last_stop_reason

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time.

        Must not be captured once at construction: with checkpointing
        active, the enclosing checkpoint span rotates at each fire and a
        frozen id would pin every event to the first checkpoint.
        """
        return current_span_id()

    def reset(self) -> None:
        """Close any open spans and clear per-attempt state.

        Called between Claude Code subprocess restarts (retry attempts) and
        in a `finally` after the retry loop. Emits `SpanEndEvent` for every
        still-open agent span (innermost first) so the transcript stays
        balanced even if Claude Code crashed before its tool_result blocks
        were written.
        """
        for tool_use_id in reversed(list(self._open_agents.keys())):
            agent = self._open_agents.pop(tool_use_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        self._pending_subagents.clear()
        self._task_ids.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        event.span_id = self._attribute(event.input)
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        # Record the terminal stop reason (guard against empty choices, where
        # ModelOutput.stop_reason raises). This fires for sub-agent calls too,
        # but the runner only consults it on a non-zero exit.
        if event.output and event.output.choices:
            self._last_stop_reason = event.output.stop_reason

        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            # Attach custom rendering for Claude Code's built-in tools.
            # inspect_ai's `tool_call_view` only handles tools registered as
            # ToolDefs; Claude Code's built-in tools (Write, Task, Agent,
            # ExitPlanMode, …) aren't, so we fill in our own views here
            # before `_event_updated` lets the viewer render the call.
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {})
                    if custom is not None:
                        tc.view = custom

            # If this call's response launched any Task/Agent sub-agents,
            # open their spans and register pending entries NOW —
            # synchronously, before the bridge response is sent back to
            # Claude Code. By the time any sub-agent makes its first bridge
            # call, both `_open_agents` and `_pending_subagents` are ready.
            parent_span_id = event.span_id or self.outer_span_id
            for tc in msg.tool_calls:
                if tc.function not in ("Task", "Agent"):
                    continue
                args = tc.arguments or {}
                prompt = args.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    continue
                if tc.id in self._launched:
                    # idempotent — defensive against retries
                    continue
                name = args.get("name")
                self._launched[tc.id] = _Launched(
                    prompt=prompt,
                    span_name=str(args.get("subagent_type") or name or "agent"),
                    description=str(args.get("description") or ""),
                    parent_span_id=parent_span_id,
                    name=name if isinstance(name, str) else None,
                    fork=args.get("subagent_type") == "fork",
                    launcher=self._origin_of(parent_span_id),
                )
                self._open_span(tc.id, tc.id)

            # SendMessage to a finished background agent resumes it: reopen
            # its span now, before the resumed run's first call arrives
            for tc in msg.tool_calls:
                if tc.function != "SendMessage":
                    continue
                origin = self._find_launched(
                    (tc.arguments or {}).get("to"), parent_span_id
                )
                if origin is None:
                    continue
                if self._is_open(origin):
                    # still running (message is delivered in-run), or its
                    # notification hasn't reached us yet
                    self._launched[origin].pending_send = tc.id
                else:
                    self._open_span(tc.id, origin, parent_span_id)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def _open_span(
        self,
        key: str,
        origin: str,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        stale_run_ids: set[str] | None = None,
    ) -> _OpenAgent:
        launched = self._launched[origin]
        resumed = launched.runs > 0
        launched.runs += 1
        launched.pending_send = None
        span_id = f"agent-{origin}-run{launched.runs}" if resumed else f"agent-{origin}"
        parent_span_id = parent_span_id or launched.parent_span_id
        agent = _OpenAgent(
            span_id=span_id,
            origin=origin,
            via_send_message=key != origin,
            run_id=run_id,
            stale_run_ids=stale_run_ids or set(),
        )
        self._open_agents[key] = agent
        launched.span_id = span_id
        self._pending_subagents[key] = launched.prompt
        if resumed and launched.agent_id:
            # a resumed run's notification may name only the task id
            self._task_ids[launched.agent_id] = key
        metadata: dict[str, str] = {}
        if launched.description:
            metadata["description"] = launched.description
        if resumed:
            metadata["resumes"] = f"agent-{origin}"
        transcript()._event(
            SpanBeginEvent(
                id=span_id,
                parent_id=parent_span_id,
                type="agent",
                name=launched.span_name,
                metadata=metadata or None,
            )
        )
        return agent

    def _is_open(self, origin: str) -> bool:
        return self._open_key(origin) is not None

    def _open_key(self, origin: str) -> str | None:
        """Key of the open span for a launched sub-agent's current run."""
        for key, agent in self._open_agents.items():
            if agent.origin == origin:
                return key
        return None

    def _origin_of(self, span_id: str | None) -> str | None:
        """The sub-agent a span belongs to (None: the main agent)."""
        for origin, launched in self._launched.items():
            if (
                span_id is not None
                and span_id
                in (
                    f"agent-{origin}",
                    launched.span_id,
                )
                or (span_id or "").startswith(f"agent-{origin}-run")
            ):
                return origin
        return None

    def _find_launched(self, ref: Any, caller_span_id: str | None = None) -> str | None:
        """Launching tool_use_id of the sub-agent a SendMessage `to` names.

        An agent id is unique; a name is only unique among one agent's
        children, so name matches prefer the caller's own launches.
        """
        if not isinstance(ref, str) or not ref:
            return None
        by_id = [o for o, la in self._launched.items() if la.agent_id == ref]
        if by_id:
            return by_id[-1]
        by_name = [o for o, la in self._launched.items() if la.name == ref]
        caller = self._origin_of(caller_span_id)
        own = [o for o in by_name if self._launched[o].launcher == caller]
        matches = own or by_name
        return matches[-1] if matches else None

    def _roll_over(self, key: str) -> _OpenAgent:
        """Close the current run's span and open the next run's."""
        agent = self._open_agents[key]
        launched = self._launched[agent.origin]
        stale = set(agent.stale_run_ids)
        if agent.run_id:
            stale.add(agent.run_id)
        new_key = launched.pending_send or key
        self._close_agent(key)
        return self._open_span(new_key, agent.origin, stale_run_ids=stale)

    def _bind_agent_id(self, origin: str, agent_id: str) -> None:
        launched = self._launched.get(origin)
        if launched is not None and not launched.agent_id:
            launched.agent_id = agent_id

    # ------------------------------------------------------------------
    # JSONL consumer
    # ------------------------------------------------------------------

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
        """Process one raw JSONL event from Claude Code.

        Called from the runner loop as each JSONL line arrives.

        Sub-agent spans open from the bridge (`on_complete`/`on_pending`,
        ahead of the race with stdout-buffered JSONL arrival). Here we
        consume `user` lines (tool_result → span close) and `system` lines
        (compaction, task lifecycle).
        """
        event_type = raw.get("type")
        if event_type == "user":
            self._handle_user(raw)
        elif event_type == "system":
            self._handle_system(raw)

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """Resolve the span_id for an incoming bridge call.

        Forks are matched first (see `_match_fork`). Otherwise
        substring-matches the first user message's text against
        currently-pending sub-agent prompts. Exactly one match → that
        sub-agent's span. Zero or multiple matches → outer span. A call
        that shows a run boundary we have no span for yet starts the next
        run's span (a resumed agent whose resume we have not seen).
        """
        fork = self._match_fork(input_messages)
        if fork is not None:
            origin, start = fork
            return self._run_span(origin, input_messages[start:])

        user_text = self._first_user_text(input_messages)
        if not user_text:
            return self.outer_span_id

        # a parent working beside a background sub-agent may quote its prompt;
        # only the parent's history holds the launching tool call
        launched_here = {
            tc.id
            for msg in input_messages
            if isinstance(msg, ChatMessageAssistant)
            for tc in msg.tool_calls or []
        }
        boundaries = _run_boundaries(input_messages)
        matches = [
            origin
            for origin, launched in self._launched.items()
            if not launched.fork
            and len(launched.prompt) >= _MIN_PROMPT_LENGTH
            and launched.prompt in user_text
            and origin not in launched_here
            # a finished agent only matches a call that shows a resume
            and (self._is_open(origin) or boundaries >= 1)
        ]
        best = self._best_match(matches, boundaries)
        if best is None:
            return self.outer_span_id
        return self._run_span(best, input_messages)

    def _best_match(self, matches: list[str], boundaries: int) -> str | None:
        """Rank prompt matches (several when prompts repeat or nest).

        The longest prompt wins over prompts it contains; then an open agent
        whose current run fits the call; then a finished agent the call
        resumes; then the most recent launch.
        """
        prompts = {o: self._launched[o].prompt for o in matches}
        matches = [
            o
            for o in matches
            if not any(prompts[o] != p and prompts[o] in p for p in prompts.values())
        ]
        if len(matches) <= 1:
            return matches[0] if matches else None

        def rank(origin: str) -> tuple[int, int]:
            runs = self._launched[origin].runs
            if self._is_open(origin) and boundaries == runs - 1:
                tier = 2
            elif boundaries >= runs:
                tier = 1
            else:
                tier = 0
            return (tier, list(self._launched).index(origin))

        return max(matches, key=rank)

    def _run_span(self, origin: str, history: list[ChatMessage]) -> str | None:
        """Span for a call from a launched sub-agent.

        `history` is the sub-agent's own history. If it shows more finished
        runs than we have spans for, this call starts the next run (a resume
        we have not seen yet, or one that raced its predecessor's close).
        """
        launched = self._launched[origin]
        new_run = _run_boundaries(history) >= launched.runs
        key = self._open_key(origin)
        if key is not None:
            return (self._roll_over(key) if new_run else self._open_agents[key]).span_id
        if new_run:
            return self._open_span(launched.pending_send or origin, origin).span_id
        return launched.span_id

    def _match_fork(self, input_messages: list[ChatMessage]) -> tuple[str, int] | None:
        """(launch tool_use_id, directive index) if this call is from a fork.

        Claude Code gives every tool_use in the launching message the same
        "Fork started" placeholder, then the fork's directive (its prompt) in
        one user message. The last such group is the innermost fork; among
        its fork launches, the directive picks which one this is.
        """
        last = max(
            (
                i
                for i, m in enumerate(input_messages)
                if isinstance(m, ChatMessageTool)
                and m.text.lstrip().startswith("Fork started")
            ),
            default=None,
        )
        if last is None or last + 1 >= len(input_messages):
            return None
        directive = input_messages[last + 1]
        if not isinstance(directive, ChatMessageUser):
            return None
        group: list[str] = []
        i = last
        while i >= 0 and isinstance(input_messages[i], ChatMessageTool):
            group.append(str(cast(ChatMessageTool, input_messages[i]).tool_call_id))
            i -= 1
        forks = [o for o in group if o in self._launched and self._launched[o].fork]
        if len(forks) > 1:
            forks = [o for o in forks if self._launched[o].prompt in directive.text]
        return (forks[0], last + 1) if len(forks) == 1 else None

    @staticmethod
    def _first_user_text(input_messages: list[ChatMessage]) -> str | None:
        """Return the text of the first ChatMessageUser past leading system messages."""
        for msg in input_messages:
            if isinstance(msg, ChatMessageSystem):
                continue
            if isinstance(msg, ChatMessageUser):
                return msg.text
            break
        return None

    # ------------------------------------------------------------------
    # JSONL event handlers
    # ------------------------------------------------------------------

    def _handle_user(self, raw: dict[str, Any]) -> None:
        """Close agent spans and clear pending-sub-agent entries for tool_result blocks."""
        message = raw.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return

        blocks = [
            b
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "tool_result"
            and b.get("tool_use_id")
        ]
        # `tool_use_result` describes one tool's output, so only trust it
        # when it can belong to just this block
        structured = raw.get("tool_use_result") if len(blocks) == 1 else None
        for block in blocks:
            tool_use_id = str(block["tool_use_id"])
            if tool_use_id in self._launched:
                agent_id = _text_agent_id(block)
                if agent_id:
                    self._bind_agent_id(tool_use_id, agent_id)
            agent = self._open_agents.get(tool_use_id)
            if agent is not None:
                if agent.via_send_message:
                    if _send_message_succeeded(structured, block):
                        continue
                    self._close_agent(tool_use_id)
                    continue
                task_id = _async_launch_task_id(
                    structured, block, self._pending_subagents.get(tool_use_id)
                )
                if task_id is not None:
                    # background agent: still running, close on task_notification
                    if task_id:
                        self._task_ids[task_id] = tool_use_id
                        self._bind_agent_id(agent.origin, task_id)
                    continue
            self._close_agent(tool_use_id)

    def _close_agent(self, tool_use_id: str) -> None:
        # Clear pending-subagent entry (no-op for non-Task tool_results).
        self._pending_subagents.pop(tool_use_id, None)
        agent = self._open_agents.pop(tool_use_id, None)
        if agent is not None:
            self._task_ids = {
                k: v for k, v in self._task_ids.items() if v != tool_use_id
            }
            transcript()._event(SpanEndEvent(id=agent.span_id))

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Handle compaction boundaries and background-agent completion.

        Spans open from the bridge (synchronous with the calls, ahead of any
        race). `task_started` binds the task and run ids, and opens a span
        only for a resumed run the bridge has not shown us yet.
        `task_notification` closes background agents, whose `tool_result`
        arrived at launch; foreground agents close in `_handle_user`. A late
        notification for a run we already rolled past is ignored.
        """
        subtype = raw.get("subtype")
        tool_use_id = raw.get("tool_use_id")
        task_id = raw.get("task_id")
        run_id = raw.get("run_id")
        run_id = run_id if isinstance(run_id, str) else None
        if subtype == "compact_boundary":
            self._handle_compact_boundary(raw)
        elif subtype == "task_started" and isinstance(task_id, str):
            if isinstance(tool_use_id, str) and tool_use_id in self._launched:
                self._bind_agent_id(tool_use_id, task_id)
            origin = self._find_launched(task_id)
            if origin is None or not isinstance(tool_use_id, str):
                return
            open_key = self._open_key(origin)
            if open_key is None:
                self._open_span(tool_use_id, origin, run_id=run_id)
                return
            current = self._open_agents[open_key]
            if current.run_id is None:
                current.run_id = run_id
            elif run_id is not None and run_id != current.run_id:
                # a new run we have not seen from the bridge yet
                self._roll_over(open_key).run_id = run_id
        elif subtype == "task_notification":
            if not isinstance(tool_use_id, str) or tool_use_id not in self._open_agents:
                tool_use_id = (
                    self._task_ids.get(task_id) if isinstance(task_id, str) else None
                )
            if tool_use_id is None:
                return
            agent = self._open_agents.get(tool_use_id)
            if agent is not None and run_id in agent.stale_run_ids:
                return
            self._close_agent(tool_use_id)

    def _handle_compact_boundary(self, raw: dict[str, Any]) -> None:
        parent_tool_use_id = raw.get("parent_tool_use_id")
        if parent_tool_use_id:
            parent_agent = self._open_agents.get(parent_tool_use_id)
            span_id = parent_agent.span_id if parent_agent else self.outer_span_id
        else:
            span_id = self.outer_span_id

        compact_meta = raw.get("compactMetadata") or {}
        transcript()._event(
            CompactionEvent(
                source="claude_code",
                tokens_before=compact_meta.get("preTokens"),
                span_id=span_id,
                metadata={
                    "trigger": compact_meta.get("trigger", "auto"),
                    "content": raw.get("content") or "Conversation compacted",
                },
            )
        )


def _async_launch_task_id(
    structured: Any, block: dict[str, Any], prompt: str | None
) -> str | None:
    """Task id ("" if unknown) if a Task/Agent tool_result is a background launch ack.

    Returns None if it is not one: the sub-agent has finished. Only
    `async_launched` agents keep running locally; `remote_launched` agents run
    in the cloud and never call the bridge, so their span closes at launch.
    Prefers the structured `tool_use_result` Claude Code attaches to the JSONL
    `user` line (checked against the prompt it echoes); falls back to the
    result text.
    """
    if (
        isinstance(structured, dict)
        and "status" in structured
        and structured.get("prompt") == prompt
    ):
        if structured["status"] != "async_launched":
            return None
        return str(structured.get("agentId") or "")
    content = block.get("content")
    if isinstance(content, list):
        content = next(
            (str(c.get("text", "")) for c in content if isinstance(c, dict)), ""
        )
    if isinstance(content, str) and content.lstrip().startswith("Async agent launched"):
        return ""
    return None


def _run_boundaries(history: list[ChatMessage]) -> int:
    """Finished runs visible in a sub-agent's history.

    A run ends with a text-only assistant turn; a resumed run continues the
    same history with a new user turn after it. Claude Code can also add a
    user turn inside one run (max-output-tokens recovery, a blocking
    SubagentStop hook); that splits the run's span in two, and the second
    stays open until `reset()`.
    """
    return sum(
        1
        for prev, nxt in zip(history, history[1:], strict=False)
        if isinstance(prev, ChatMessageAssistant)
        and not prev.tool_calls
        and isinstance(nxt, ChatMessageUser)
    )


def _block_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ""


_AGENT_ID_LINE = re.compile(r"^agentId: (\S+)", re.MULTILINE)


def _text_agent_id(block: dict[str, Any]) -> str | None:
    """The `agentId: …` line Claude Code puts in Task/Agent results."""
    m = _AGENT_ID_LINE.search(_block_text(block))
    return m.group(1) if m else None


def _send_message_succeeded(structured: Any, block: dict[str, Any]) -> bool:
    """Whether a SendMessage result says the agent is being resumed."""
    if block.get("is_error"):
        return False
    if isinstance(structured, dict):
        return structured.get("success") is True or any(
            k in structured for k in ("resumedAgentId", "revivedAgentId")
        )
    text = _block_text(block)
    return '"success":true' in text.replace(" ", "") or "Resuming agent" in text
