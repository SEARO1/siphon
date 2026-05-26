from typing import Optional
import os

from livekit.plugins import openai
from . import ClientWrapperMixin

class LLM(ClientWrapperMixin):
    """MiniMax-backed LLM wrapper for LiveKit's OpenAI-compatible plugin."""

    DEFAULT_BASE_URL = "https://api.minimax.chat/v1"

    def __init__(
        self,
        model: Optional[str] = "MiniMax-M2.7",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        group_id: Optional[str] = None,  # MiniMax requires group_id
        temperature: Optional[float] = 0.3,
        max_completion_tokens: Optional[int] = 150,
        parallel_tool_calls: Optional[bool] = True,
        timeout: Optional[int] = 15,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("MINIMAX_API_KEY")

        if not api_key:
            raise ValueError("MINIMAX_API_KEY environment variable is not set")

        self.model = model
        self.base_url = base_url or os.getenv("MINIMAX_BASE_URL", self.DEFAULT_BASE_URL)
        self.api_key = api_key
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID")
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.parallel_tool_calls = parallel_tool_calls
        self.timeout = timeout

        self._client = self._build_client()

    def _build_client(self):
        return openai.LLM(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            parallel_tool_calls=self.parallel_tool_calls,
            timeout=self.timeout,
        )

    # JSON-serializable view (no Python objects)
    def to_config(self) -> dict:
        return {
            "provider": "minimax",
            "model": self.model,
            "base_url": self.base_url,
            "group_id": self.group_id,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "parallel_tool_calls": self.parallel_tool_calls,
            "timeout": self.timeout,
        }

    # Recreate LLM from config dict
    @classmethod
    def from_config(cls, cfg: dict) -> "LLM":
        return cls(
            model=cfg.get("model", "MiniMax-Text-01"),
            base_url=cfg.get("base_url"),
            group_id=cfg.get("group_id"),
            temperature=cfg.get("temperature", 0.3),
            max_completion_tokens=cfg.get("max_completion_tokens", 150),
            parallel_tool_calls=cfg.get("parallel_tool_calls", True),
            timeout=cfg.get("timeout", 15),
        )
