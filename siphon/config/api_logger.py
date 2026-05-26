"""
api_logger.py
=============
Records and prints to the terminal the response times (latency) of each API call
made during a voice-agent session: STT (Speech-to-Text), LLM (Language Model),
and TTS (Text-to-Speech).

Usage (in entrypoint or agent code):
    from siphon.config.api_logger import APILatencyLogger
    api_logger = APILatencyLogger()
    api_logger.attach(session)   # session is a LiveKit AgentSession
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger("api-latency-logger")

# ── Force UTF-8 output on Windows so box-drawing/ANSI chars are safe ─────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Python 3.7+
    except (AttributeError, OSError):
        pass

# Disable colours when stdout is not a real terminal (redirect / CI)
_USE_COLOUR = sys.stdout.isatty()


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyRecord:
    """Single timing record for one API call."""
    service: str          # "STT" | "LLM" | "TTS"
    provider: str         # e.g. "deepgram", "openai", "cartesia"
    model: str            # model name / identifier
    latency_ms: float     # round-trip latency in milliseconds
    timestamp: float      # Unix epoch when measurement was completed
    extra: dict = field(default_factory=dict)  # optional metadata

    def __str__(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return (
            f"[{ts}] [{self.service:>3}] provider={self.provider:<12} "
            f"model={self.model:<30} latency={self.latency_ms:>8.1f} ms"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ──────────────────────────────────────────────────────────────────────────────

if _USE_COLOUR:
    _COLOURS = {
        "STT": "\033[36m",   # cyan
        "LLM": "\033[33m",   # yellow
        "TTS": "\033[35m",   # magenta
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "GREEN": "\033[32m",
        "RED": "\033[31m",
    }
else:
    _COLOURS = {k: "" for k in ("STT", "LLM", "TTS", "RESET", "BOLD", "GREEN", "RED")}

def _coloured(service: str, text: str) -> str:
    colour = _COLOURS.get(service, "")
    return f"{colour}{text}{_COLOURS['RESET']}"


# ──────────────────────────────────────────────────────────────────────────────
# Context managers – wraps a single timed API call
# ──────────────────────────────────────────────────────────────────────────────

class _Timer:
    """Simple context-manager / manual timer."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def start(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started")
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return self.elapsed_ms

    def __enter__(self) -> "_Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Main logger class
# ──────────────────────────────────────────────────────────────────────────────

class APILatencyLogger:
    """
    Measures and prints API latency for STT, LLM, and TTS calls.

    Two modes of use:
    1. **Automatic** – call ``attach(session)`` with a LiveKit AgentSession.
       The logger subscribes to session metric events and prints results
       automatically.
    2. **Manual** – call ``record(service, provider, model, latency_ms)``
       directly from your own timing code.
    """

    def __init__(self, print_summary_every: int = 10) -> None:
        """
        Args:
            print_summary_every: Print a rolling stats summary after every N
                                  records (0 = disable summaries).
        """
        self._records: List[LatencyRecord] = []
        self._print_summary_every = print_summary_every
        self._attached_session = None
        # Track which metrics have already been logged via conversation_item_added
        # so the legacy metrics_collected handler doesn't double-count them.
        self._logged_via_item: set = set()

        self._print_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        service: str,
        provider: str,
        model: str,
        latency_ms: float,
        **extra,
    ) -> LatencyRecord:
        """Manually record and print a latency measurement."""
        rec = LatencyRecord(
            service=service.upper(),
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            timestamp=time.time(),
            extra=extra,
        )
        self._records.append(rec)
        self._print_record(rec)

        if self._print_summary_every and len(self._records) % self._print_summary_every == 0:
            self.print_summary()

        return rec

    def attach(self, session) -> None:
        """
        Attach to a LiveKit ``AgentSession`` and listen for metric events.

        Two complementary event sources are used:
        - ``metrics_collected`` (deprecated but still works) – per-API-call
          timing emitted by each plugin (STT/LLM/TTS) individually.
        - ``conversation_item_added`` (modern API) – per-turn ``MetricsReport``
          attached to each ``ChatMessage``; provides ``llm_node_ttft``,
          ``tts_node_ttfb``, ``transcription_delay``, and ``e2e_latency``.
        """
        self._attached_session = session
        # Primary source: per-turn metrics on ChatMessage
        session.on("conversation_item_added", self._on_conversation_item_added)
        # Secondary / legacy source: raw plugin metrics (still emitted)
        session.on("metrics_collected", self._on_metrics_collected)
        logger.info("APILatencyLogger attached to AgentSession")

    def print_summary(self) -> None:
        """Print per-service average/min/max statistics."""
        if not self._records:
            print("\n  [APILatencyLogger] No records yet.\n")
            return

        sep = "-" * 70
        services = ["STT", "LLM", "TTS"]
        print(f"\n{_COLOURS['BOLD']}{sep}{_COLOURS['RESET']}")
        print(f"{_COLOURS['BOLD']}  API Latency Summary  (total records: {len(self._records)}){_COLOURS['RESET']}")
        print(f"{_COLOURS['BOLD']}{sep}{_COLOURS['RESET']}")

        for svc in services:
            bucket = [r.latency_ms for r in self._records if r.service == svc]
            if not bucket:
                continue
            avg = sum(bucket) / len(bucket)
            print(
                f"  {_coloured(svc, f'{svc:>3}')}  "
                f"count={len(bucket):>4}  "
                f"avg={avg:>8.1f} ms  "
                f"min={min(bucket):>8.1f} ms  "
                f"max={max(bucket):>8.1f} ms"
            )

        print(f"{_COLOURS['BOLD']}{sep}{_COLOURS['RESET']}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        sep = "-" * 70
        print(
            f"\n{_COLOURS['BOLD']}{sep}\n"
            f"  API Latency Logger  -  recording STT / LLM / TTS response times\n"
            f"{sep}{_COLOURS['RESET']}\n"
        )

    def _print_record(self, rec: LatencyRecord) -> None:
        line = str(rec)
        # Colour-code by service
        coloured_service = _coloured(rec.service, f"[{rec.service}]")
        # Rebuild line with coloured service tag
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.timestamp))
        latency_colour = (
            _COLOURS["GREEN"] if rec.latency_ms < 500
            else _COLOURS["RED"] if rec.latency_ms > 2000
            else ""
        )
        print(
            f"[{ts}] {coloured_service} "
            f"provider={rec.provider:<12} "
            f"model={rec.model:<30} "
            f"latency={latency_colour}{rec.latency_ms:>8.1f} ms{_COLOURS['RESET']}"
        )

    # ------------------------------------------------------------------
    # LiveKit event handlers
    # ------------------------------------------------------------------

    def _on_metrics_collected(self, event) -> None:
        """
        Handle LiveKit ``metrics_collected`` (legacy, still emitted).

        Only fires for STT audio_duration when the primary
        ``conversation_item_added`` handler has not already captured STT.
        LLM and TTS are always captured via ``conversation_item_added`` in
        livekit-agents ≥ 1.5, so we skip them here to avoid duplication.
        """
        # Unwrap the event wrapper: the actual metric lives at event.metrics
        raw = self._safe_attr(event, "metrics") or event
        cls_name = type(raw).__name__.upper()

        # ── STT only (fallback) ──
        # For streaming STT (Deepgram etc.), duration == 0.0.
        # Use audio_duration as a proxy for processing time.
        if "STT" in cls_name:
            # Skip if conversation_item_added already recorded STT this turn
            req_id = self._safe_attr(raw, "request_id") or ""
            if f"stt_raw_{req_id}" in self._logged_via_item:
                return
            # Prefer streamed audio_duration (> 0 always); duration == 0 for streaming
            audio_dur_s = self._safe_attr(raw, "audio_duration") or 0.0
            duration_s  = self._safe_attr(raw, "duration") or 0.0
            latency_s   = duration_s if duration_s > 0 else audio_dur_s
            if latency_s and latency_s > 0:
                provider = self._safe_attr(raw, "label") or "stt"
                model    = self._get_model(raw)
                self.record("STT[raw]", provider, model, latency_s * 1000)
                self._logged_via_item.add(f"stt_raw_{req_id}")

    def _on_conversation_item_added(self, event) -> None:
        """
        Handle ``conversation_item_added`` events (modern LiveKit API).

        Each ``ChatMessage`` carries a ``MetricsReport`` dict with per-turn
        latency fields:
          User messages  → transcription_delay, stt_metadata
          Asst messages  → llm_node_ttft, tts_node_ttfb, e2e_latency,
                           llm_metadata, tts_metadata

        This is the PRIMARY source of per-turn latency in livekit-agents 1.5+.
        The ``metrics_collected`` handler is kept as a fallback for STT audio
        duration but skips STT/LLM/TTS if this handler has already fired.
        """
        item = self._safe_attr(event, "item")
        if item is None:
            return

        item_type = type(item).__name__
        if item_type != "ChatMessage":
            return   # skip AgentHandoff or unknown types

        role    = self._safe_attr(item, "role")
        metrics = self._safe_attr(item, "metrics")
        if not metrics:
            return

        item_id = self._safe_attr(item, "id") or id(item)

        if role == "user":
            # STT / transcription pipeline latency
            delay_s = metrics.get("transcription_delay")
            if delay_s and delay_s > 0:
                stt_meta = metrics.get("stt_metadata") or {}
                provider = stt_meta.get("model_provider") or "stt"
                model    = stt_meta.get("model_name") or ""
                self.record("STT", provider, model, delay_s * 1000)
                self._logged_via_item.add(f"stt_{item_id}")

        elif role == "assistant":
            # LLM time-to-first-token
            llm_ttft_s = metrics.get("llm_node_ttft")
            if llm_ttft_s and llm_ttft_s > 0:
                llm_meta = metrics.get("llm_metadata") or {}
                provider = llm_meta.get("model_provider") or "llm"
                model    = llm_meta.get("model_name") or ""
                self.record("LLM", provider, model, llm_ttft_s * 1000)
                self._logged_via_item.add(f"llm_{item_id}")

            # TTS time-to-first-byte
            tts_ttfb_s = metrics.get("tts_node_ttfb")
            if tts_ttfb_s and tts_ttfb_s > 0:
                tts_meta = metrics.get("tts_metadata") or {}
                provider = tts_meta.get("model_provider") or "tts"
                model    = tts_meta.get("model_name") or ""
                self.record("TTS", provider, model, tts_ttfb_s * 1000)
                self._logged_via_item.add(f"tts_{item_id}")

            # E2E latency (user finished speaking → agent started speaking)
            e2e_s = metrics.get("e2e_latency")
            if e2e_s and e2e_s > 0:
                ts_now = time.time()
                rec = LatencyRecord(
                    service="E2E",
                    provider="pipeline",
                    model="end-to-end",
                    latency_ms=e2e_s * 1000,
                    timestamp=ts_now,
                )
                self._records.append(rec)
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_now))
                colour = _COLOURS.get("LLM", "")  # yellow for e2e
                print(
                    f"[{ts_str}] {colour}[E2E]{_COLOURS['RESET']} "
                    f"pipeline end-to-end latency = "
                    f"{colour}{e2e_s * 1000:>8.1f} ms{_COLOURS['RESET']}"
                )

    @staticmethod
    def _safe_attr(obj, name: str):
        """Return attribute value or None if missing."""
        return getattr(obj, name, None)

    @staticmethod
    def _get_model(raw) -> str:
        """Extract model name from a raw metrics object."""
        meta = getattr(raw, "metadata", None)
        if meta is not None:
            return getattr(meta, "model_name", None) or ""
        return ""

    @staticmethod
    def _extract_ms(obj, name: str) -> Optional[float]:
        """
        Return a metric field converted to milliseconds.

        LiveKit metric durations are in seconds (float).  Multiply by 1000.
        Returns None if the field is missing or zero.
        """
        val = getattr(obj, name, None)
        if val is None:
            return None
        if isinstance(val, (int, float)) and val > 0:
            return float(val) * 1000  # seconds → ms
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Convenience context managers for manual timing
# ──────────────────────────────────────────────────────────────────────────────

class timed_stt:
    """Context manager that times an STT call and records it."""

    def __init__(self, logger_instance: APILatencyLogger, provider: str, model: str = ""):
        self._logger = logger_instance
        self._provider = provider
        self._model = model
        self._timer = _Timer()

    def __enter__(self):
        self._timer.start()
        return self

    def __exit__(self, exc_type, *_):
        ms = self._timer.stop()
        if exc_type is None:
            self._logger.record("STT", self._provider, self._model, ms)


class timed_llm:
    """Context manager that times an LLM call and records it."""

    def __init__(self, logger_instance: APILatencyLogger, provider: str, model: str = ""):
        self._logger = logger_instance
        self._provider = provider
        self._model = model
        self._timer = _Timer()

    def __enter__(self):
        self._timer.start()
        return self

    def __exit__(self, exc_type, *_):
        ms = self._timer.stop()
        if exc_type is None:
            self._logger.record("LLM", self._provider, self._model, ms)


class timed_tts:
    """Context manager that times a TTS call and records it."""

    def __init__(self, logger_instance: APILatencyLogger, provider: str, model: str = ""):
        self._logger = logger_instance
        self._provider = provider
        self._model = model
        self._timer = _Timer()

    def __enter__(self):
        self._timer.start()
        return self

    def __exit__(self, exc_type, *_):
        ms = self._timer.stop()
        if exc_type is None:
            self._logger.record("TTS", self._provider, self._model, ms)
