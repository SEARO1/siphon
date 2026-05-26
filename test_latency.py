"""
API Call Latency Tester
=======================
Tests the first-token / first-audio-chunk latency for:
  - LLM  : Azure OpenAI (chat completions, streaming)
  - STT  : Deepgram (pre-recorded transcription REST endpoint)
  - TTS  : Cartesia  (streaming TTS, time-to-first-audio-chunk)

Usage:
    python test_latency.py [--runs N]

Results are printed to stdout as a simple table and also saved to
latency_results.json in the working directory.
"""

import argparse
import asyncio
import io
import json
import os
import time
import wave
from typing import Optional

import aiohttp
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_BASE_URL", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL: str = "nova-3"
DEEPGRAM_LANGUAGE: str = "en"

CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_MODEL: str = "sonic-3"
CARTESIA_VOICE_ID: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"

# Test inputs
LLM_TEST_PROMPT = "Reply with exactly one short sentence: what is 1+1?"
TTS_TEST_TEXT = "Hello, this is a latency test for the text to speech API."

# A minimal 1-second silent 16-kHz mono WAV used as STT input
def _make_silent_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    n_frames = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


SILENT_WAV_BYTES = _make_silent_wav()

# ─── Azure OpenAI LLM latency (streaming, TTFT) ───────────────────────────────

def measure_azure_llm_latency(prompt: str = LLM_TEST_PROMPT) -> dict:
    """
    Sends a streaming chat completion request and measures:
      - total_ms  : wall-clock time until the HTTP response is fully consumed
      - ttft_ms   : time-to-first-token (first SSE data chunk received)
    """
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "stream": True,
    }

    ttft_ms: Optional[float] = None
    t_start = time.perf_counter()

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line != b"data: [DONE]":
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                    # Keep consuming until done so total_ms is accurate
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "service": "Azure OpenAI LLM",
            "status": "ok",
            "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
            "total_ms": round(total_ms, 2),
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "service": "Azure OpenAI LLM",
            "status": f"error: {exc}",
            "ttft_ms": None,
            "total_ms": round(total_ms, 2),
        }


# ─── Deepgram STT latency (pre-recorded REST) ─────────────────────────────────

def measure_deepgram_stt_latency() -> dict:
    """
    Uploads a short silent WAV to Deepgram's pre-recorded transcription endpoint
    and measures how long the round-trip takes.

    Note: for a fair latency comparison use a real speech WAV; silent audio will
    still exercise the API's network + processing path.
    """
    url = (
        f"https://api.deepgram.com/v1/listen"
        f"?model={DEEPGRAM_MODEL}&language={DEEPGRAM_LANGUAGE}"
        f"&smart_format=true&punctuate=true"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    t_start = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, data=SILENT_WAV_BYTES, timeout=30)
        total_ms = (time.perf_counter() - t_start) * 1000
        resp.raise_for_status()
        result = resp.json()
        transcript = (
            result.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        return {
            "service": "Deepgram STT",
            "status": "ok",
            "total_ms": round(total_ms, 2),
            "transcript_preview": transcript[:80] or "(empty – silent audio)",
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "service": "Deepgram STT",
            "status": f"error: {exc}",
            "total_ms": round(total_ms, 2),
        }


# ─── Cartesia TTS latency (streaming, TTFA) ───────────────────────────────────

async def _measure_cartesia_tts_async(text: str = TTS_TEST_TEXT) -> dict:
    """
    Sends a streaming TTS request to Cartesia and measures:
      - ttfa_ms   : time-to-first-audio-chunk
      - total_ms  : wall-clock time for the full response
    """
    url = "https://api.cartesia.ai/tts/bytes"
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json",
    }
    payload = {
        "model_id": CARTESIA_MODEL,
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": CARTESIA_VOICE_ID,
        },
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
        },
        "stream": True,
    }

    ttfa_ms: Optional[float] = None
    t_start = time.perf_counter()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk and ttfa_ms is None:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "service": "Cartesia TTS",
            "status": "ok",
            "ttfa_ms": round(ttfa_ms, 2) if ttfa_ms else None,
            "total_ms": round(total_ms, 2),
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "service": "Cartesia TTS",
            "status": f"error: {exc}",
            "ttfa_ms": None,
            "total_ms": round(total_ms, 2),
        }


def measure_cartesia_tts_latency() -> dict:
    return asyncio.run(_measure_cartesia_tts_async())


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all(runs: int = 3) -> None:
    all_results: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  API Latency Test  --  {runs} run(s) per service")
    print(f"{'='*60}\n")

    services = [
        ("Azure OpenAI LLM (streaming TTFT)", measure_azure_llm_latency),
        ("Deepgram STT (pre-recorded REST)",  measure_deepgram_stt_latency),
        ("Cartesia TTS (streaming TTFA)",      measure_cartesia_tts_latency),
    ]

    for label, fn in services:
        print(f">>  {label}")
        run_results = []
        for i in range(1, runs + 1):
            result = fn()
            run_results.append(result)
            _print_run(i, result)
        _print_summary(run_results)
        all_results.extend(run_results)
        print()

    # Save to JSON
    out_path = "latency_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _print_run(idx: int, r: dict) -> None:
    status = r.get("status", "?")
    parts = [f"  Run {idx}: status={status}"]
    for key in ("ttft_ms", "ttfa_ms", "total_ms"):
        if key in r and r[key] is not None:
            parts.append(f"{key}={r[key]:.1f} ms")
    if "transcript_preview" in r:
        parts.append(f'transcript="{r["transcript_preview"]}"')
    print("  |  ".join(parts))


def _print_summary(results: list[dict]) -> None:
    def _safe_avg(key: str) -> Optional[float]:
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    print(f"  {'-'*48}")
    for key in ("ttft_ms", "ttfa_ms", "total_ms"):
        avg = _safe_avg(key)
        if avg is not None:
            print(f"  avg {key:12s}: {avg:.1f} ms")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure LLM / STT / TTS API latency")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of test runs per service (default: 3)",
    )
    args = parser.parse_args()

    # Quick credential check
    missing = []
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_BASE_URL")
    if not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")
    if not CARTESIA_API_KEY:
        missing.append("CARTESIA_API_KEY")

    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("        Please set them in .env or export them before running.")
        raise SystemExit(1)

    run_all(runs=args.runs)
