"""Unit tests for streaming display components.

Covers:
  - _render_stream_text()   helper for think-tag styling
  - _StreamingPanel         live Rich renderable
  - DisplayManager          token routing and panel lifecycle
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from rich.panel import Panel
from rich.text import Text

from backend.api.schemas.pipeline import StreamEvent, StreamEventType
from cli.display import (
    AGENT_STYLE,
    DisplayManager,
    _StreamingPanel,
    _render_stream_text,
    _STREAM_MAX_LINES,
)


# ---------------------------------------------------------------------------
# _render_stream_text
# ---------------------------------------------------------------------------

class TestRenderStreamText:
    def test_empty_inputs_return_empty_text(self):
        result = _render_stream_text("", "")
        assert isinstance(result, Text)
        assert result.plain == ""

    def test_plain_text_rendered(self):
        text = "hello world"
        result = _render_stream_text(text, text)
        assert "hello world" in result.plain

    def test_plain_text_uses_dim_style(self):
        text = "some code"
        result = _render_stream_text(text, text)
        # The whole text should be rendered (dim style = dim #cdd6f4)
        assert result.plain == text

    def test_think_block_isolated_from_rest(self):
        full = "<think>hidden reasoning</think>visible output"
        result = _render_stream_text(full, full)
        # Both segments must appear in the plain text
        assert "hidden reasoning" in result.plain
        assert "visible output" in result.plain

    def test_think_content_has_different_spans(self):
        full = "<think>reason</think>code"
        result = _render_stream_text(full, full)
        # Rich Text stores spans; verify there are at least 2 distinct spans
        assert len(result._spans) >= 2

    def test_unclosed_think_tag_treated_as_ongoing(self):
        full = "<think>still reasoning..."
        result = _render_stream_text(full, full)
        # All content after <think> is think-mode — should appear in output
        assert "still reasoning..." in result.plain

    def test_visible_window_starts_inside_think_block(self):
        """Prefix establishes we're already inside a <think> block."""
        prefix = "<think>" + "x" * 200 + "\n" * 15   # 15 lines in think
        suffix = "last line of think</think>\nnormal"
        full = prefix + suffix

        # visible_text = last few lines of full
        lines = full.splitlines()
        visible_lines = lines[-5:]
        visible = "\n".join(visible_lines)

        result = _render_stream_text(full, visible)
        assert result.plain.strip() != ""

    def test_multiple_think_blocks(self):
        full = "<think>r1</think>out1<think>r2</think>out2"
        result = _render_stream_text(full, full)
        assert "r1" in result.plain
        assert "out1" in result.plain
        assert "r2" in result.plain
        assert "out2" in result.plain

    def test_visible_text_subset_of_full(self):
        """When visible_text is a suffix of full_text, only that portion is rendered."""
        full = "line1\nline2\nline3\nline4\nline5"
        visible = "line4\nline5"
        result = _render_stream_text(full, visible)
        assert "line4" in result.plain
        assert "line5" in result.plain
        assert "line1" not in result.plain


# ---------------------------------------------------------------------------
# _StreamingPanel
# ---------------------------------------------------------------------------

class TestStreamingPanel:
    def test_init_known_agent(self):
        panel = _StreamingPanel("actor")
        name, color = AGENT_STYLE["actor"]
        assert panel._name == name
        assert panel._color == color
        assert panel._chunks == []

    def test_init_unknown_agent_uses_title_case(self):
        panel = _StreamingPanel("my_custom_agent")
        assert panel._name == "My Custom Agent"

    def test_add_token_appends(self):
        panel = _StreamingPanel("actor")
        panel.add_token("hello")
        panel.add_token(" world")
        assert panel._chunks == ["hello", " world"]

    def test_rich_returns_panel(self):
        panel = _StreamingPanel("actor")
        renderable = panel.__rich__()
        assert isinstance(renderable, Panel)

    def test_rich_panel_has_agent_border_style(self):
        panel = _StreamingPanel("checker")
        _, color = AGENT_STYLE["checker"]
        renderable = panel.__rich__()
        assert renderable.border_style == color

    def test_rich_with_no_tokens_renders_empty_body(self):
        panel = _StreamingPanel("actor")
        renderable = panel.__rich__()
        # Panel's renderable (the body) should be an empty or near-empty Text
        # We just verify it doesn't crash and is a Panel
        assert isinstance(renderable, Panel)

    def test_rich_body_contains_streamed_text(self):
        panel = _StreamingPanel("actor")
        panel.add_token("def ")
        panel.add_token("f():")
        panel.add_token(" return 1")
        renderable = panel.__rich__()
        # The body is a Text object; its plain content must include streamed tokens
        body: Text = renderable.renderable
        assert "def " in body.plain
        assert "f():" in body.plain

    def test_rich_title_contains_agent_name(self):
        panel = _StreamingPanel("policy")
        name, _ = AGENT_STYLE["policy"]
        renderable = panel.__rich__()
        assert name in renderable.title.plain

    def test_rich_truncates_to_max_lines(self):
        panel = _StreamingPanel("actor")
        # Add more lines than _STREAM_MAX_LINES
        for i in range(_STREAM_MAX_LINES + 5):
            panel.add_token(f"line{i}\n")
        renderable = panel.__rich__()
        body: Text = renderable.renderable
        visible_lines = [l for l in body.plain.splitlines() if l]
        assert len(visible_lines) <= _STREAM_MAX_LINES + 1  # +1 for last partial line

    def test_rich_timer_shows_elapsed(self):
        import time
        panel = _StreamingPanel("actor")
        # Force elapsed to be at least 0.0
        renderable = panel.__rich__()
        title_plain = renderable.title.plain
        # Title must contain 's' (e.g. "0.0s")
        assert "s" in title_plain

    def test_all_agent_keys_are_valid(self):
        """Every key in AGENT_STYLE can construct a _StreamingPanel without error."""
        for agent_key in AGENT_STYLE:
            panel = _StreamingPanel(agent_key)
            panel.add_token("test")
            _ = panel.__rich__()  # must not raise


# ---------------------------------------------------------------------------
# DisplayManager — token routing and panel lifecycle
# ---------------------------------------------------------------------------

class TestDisplayManagerStreaming:
    def _run_id(self):
        return uuid4()

    def _token_event(self, agent: str, token: str) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.AGENT_TOKEN,
            run_id=self._run_id(),
            agent=agent,
            data={"token": token},
        )

    def _start_event(self, agent: str) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.AGENT_START,
            run_id=self._run_id(),
            agent=agent,
        )

    def test_agent_token_no_op_when_no_panel(self):
        """Receiving AGENT_TOKEN before any panel starts must not raise."""
        dm = DisplayManager()
        assert dm._streaming_panel is None
        event = self._token_event("actor", "hello")
        dm._on_agent_token(event)  # must not raise

    def test_agent_token_feeds_panel(self):
        """Token is forwarded to _streaming_panel.add_token()."""
        dm = DisplayManager()
        mock_panel = MagicMock(spec=_StreamingPanel)
        dm._streaming_panel = mock_panel

        event = self._token_event("actor", "chunk_text")
        dm._on_agent_token(event)

        mock_panel.add_token.assert_called_once_with("chunk_text")

    def test_empty_token_is_not_forwarded(self):
        """Empty string token must not call add_token (avoid spurious renders)."""
        dm = DisplayManager()
        mock_panel = MagicMock(spec=_StreamingPanel)
        dm._streaming_panel = mock_panel

        event = self._token_event("actor", "")
        dm._on_agent_token(event)

        mock_panel.add_token.assert_not_called()

    def test_start_spinner_creates_streaming_panel(self):
        """_start_spinner() always creates a _StreamingPanel, not a _DynamicSpinner."""
        dm = DisplayManager()
        with patch("cli.display.Live") as mock_live_cls:
            mock_live_instance = MagicMock()
            mock_live_cls.return_value = mock_live_instance

            dm._start_spinner("actor")

        assert isinstance(dm._streaming_panel, _StreamingPanel)

    def test_start_spinner_sets_live(self):
        """_start_spinner() starts a Live display."""
        dm = DisplayManager()
        with patch("cli.display.Live") as mock_live_cls:
            mock_live_instance = MagicMock()
            mock_live_cls.return_value = mock_live_instance

            dm._start_spinner("checker")

        mock_live_instance.start.assert_called_once()

    def test_stop_spinner_clears_streaming_panel(self):
        """_stop_spinner() sets _streaming_panel to None."""
        dm = DisplayManager()
        with patch("cli.display.Live") as mock_live_cls:
            mock_live_instance = MagicMock()
            mock_live_cls.return_value = mock_live_instance
            dm._start_spinner("actor")

        assert dm._streaming_panel is not None
        dm._stop_spinner()
        assert dm._streaming_panel is None

    def test_stop_spinner_stops_live(self):
        """_stop_spinner() calls live.stop()."""
        dm = DisplayManager()
        with patch("cli.display.Live") as mock_live_cls:
            mock_live_instance = MagicMock()
            mock_live_cls.return_value = mock_live_instance
            dm._start_spinner("actor")

        dm._stop_spinner()
        mock_live_instance.stop.assert_called_once()

    def test_handle_event_routes_agent_token(self):
        """handle_event() dispatches AGENT_TOKEN to _on_agent_token."""
        dm = DisplayManager()
        called_with: list[StreamEvent] = []
        dm._on_agent_token = lambda e: called_with.append(e)

        event = self._token_event("actor", "x")
        dm.handle_event(event)

        assert len(called_with) == 1
        assert called_with[0].data["token"] == "x"

    def test_consecutive_tokens_accumulate_in_panel(self):
        """Multiple AGENT_TOKEN events accumulate in the same panel's chunk list."""
        dm = DisplayManager()
        dm._streaming_panel = _StreamingPanel("actor")  # inject real panel

        tokens = ["int ", "main", "()", " {", " return 0; }"]
        for tok in tokens:
            dm._on_agent_token(self._token_event("actor", tok))

        accumulated = "".join(dm._streaming_panel._chunks)
        assert accumulated == "".join(tokens)

    def test_start_spinner_stops_previous_live(self):
        """Starting a new spinner first stops any running one (no orphaned threads)."""
        dm = DisplayManager()
        with patch("cli.display.Live") as mock_live_cls:
            first = MagicMock()
            second = MagicMock()
            mock_live_cls.side_effect = [first, second]

            dm._start_spinner("actor")
            dm._start_spinner("checker")

        first.stop.assert_called_once()
