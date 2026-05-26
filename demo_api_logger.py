"""
demo_api_logger.py
==================
Standalone demo / smoke-test for APILatencyLogger.

Run with:
    python demo_api_logger.py

No LiveKit connection required – it uses the manual timing helpers to
simulate STT, LLM, and TTS calls and prints their latencies to the terminal.
"""

import time
import random

import importlib.util, pathlib, sys

# Load api_logger directly without triggering the livekit dep in siphon/__init__.py
_mod_path = pathlib.Path(__file__).parent / "siphon" / "config" / "api_logger.py"
_spec = importlib.util.spec_from_file_location("siphon.config.api_logger", _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
# Register under the dotted name so @dataclass can resolve cls.__module__
sys.modules["siphon.config.api_logger"] = _mod
_spec.loader.exec_module(_mod)

APILatencyLogger = _mod.APILatencyLogger
timed_stt        = _mod.timed_stt
timed_llm        = _mod.timed_llm
timed_tts        = _mod.timed_tts


def fake_stt_call(duration_s: float) -> str:
    """Simulate a Speech-to-Text API call."""
    time.sleep(duration_s)
    return "Hello, I would like to book an appointment."


def fake_llm_call(duration_s: float) -> str:
    """Simulate an LLM inference call."""
    time.sleep(duration_s)
    return "Sure! What date works best for you?"


def fake_tts_call(duration_s: float) -> bytes:
    """Simulate a Text-to-Speech synthesis call."""
    time.sleep(duration_s)
    return b"<audio bytes>"


def main():
    api_logger = APILatencyLogger(print_summary_every=5)

    print("Simulating 10 turns of STT → LLM → TTS pipeline...\n")

    providers = {
        "stt": ("deepgram",  "nova-3"),
        "llm": ("azure_openai", "gpt-4o-mini"),
        "tts": ("cartesia",  "sonic-3"),
    }

    for turn in range(1, 11):
        print(f"  ── Turn {turn} ──────────────────────────────────────────")

        # STT
        stt_latency = random.uniform(0.05, 0.45)   # 50–450 ms
        with timed_stt(api_logger, providers["stt"][0], providers["stt"][1]):
            fake_stt_call(stt_latency)

        # LLM
        llm_latency = random.uniform(0.3, 1.8)     # 300 ms – 1.8 s
        with timed_llm(api_logger, providers["llm"][0], providers["llm"][1]):
            fake_llm_call(llm_latency)

        # TTS
        tts_latency = random.uniform(0.05, 0.35)   # 50–350 ms
        with timed_tts(api_logger, providers["tts"][0], providers["tts"][1]):
            fake_tts_call(tts_latency)

    # Final summary
    api_logger.print_summary()

    # ----------------------------------------------------------------
    # Also demo the manual record() API
    # ----------------------------------------------------------------
    print("\nManual record() example:")
    api_logger.record("STT", "assemblyai", "universal",  210.4)
    api_logger.record("LLM", "openai",     "gpt-4o",     890.1)
    api_logger.record("TTS", "elevenlabs", "eleven_turbo_v2", 175.3)

    api_logger.print_summary()


if __name__ == "__main__":
    main()
