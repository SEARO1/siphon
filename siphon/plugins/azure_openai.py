from typing import Optional
import os

from livekit.plugins.openai import LLM as _LiveKitLLM
from . import ClientWrapperMixin

ERR_MISSING_API_KEY = "AZURE_OPENAI_API_KEY environment variable is not set"
ERR_MISSING_DEPLOYMENT = "azure_deployment is required for Azure OpenAI"


class LLM(ClientWrapperMixin):
    """Azure OpenAI-backed LLM using LiveKit's built-in with_azure()."""

    def __init__(
        self,
        model: Optional[str] = "gpt-4o-mini",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = "2024-10-21",
        temperature: Optional[float] = 0.3,
        max_completion_tokens: Optional[int] = 150,
        parallel_tool_calls: Optional[bool] = True,
        timeout: Optional[int] = 15,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not api_key:
            raise ValueError(ERR_MISSING_API_KEY)

        self.model = model
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", model)
        self.api_version = api_version
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.parallel_tool_calls = parallel_tool_calls
        self.timeout = timeout

        self._client = self._build_client()

    def _build_client(self):
        return _LiveKitLLM.with_azure(
            model=self.model,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            parallel_tool_calls=self.parallel_tool_calls,
            timeout=self.timeout,
        )

    # JSON-serializable view (no Python objects)
    def to_config(self) -> dict:
        return {
            "provider": "azure_openai",
            "model": self.model,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "parallel_tool_calls": self.parallel_tool_calls,
            "timeout": self.timeout,
        }

    # Recreate LLM from config dict
    @classmethod
    def from_config(cls, cfg: dict) -> "LLM":
        return cls(
            model=cfg.get("model", "gpt-4o-mini"),
            api_key=cfg.get("api_key"),
            azure_endpoint=cfg.get("azure_endpoint"),
            azure_deployment=cfg.get("azure_deployment"),
            api_version=cfg.get("api_version", "2024-02-01"),
            temperature=cfg.get("temperature", 0.3),
            max_completion_tokens=cfg.get("max_completion_tokens", 150),
            parallel_tool_calls=cfg.get("parallel_tool_calls", True),
            timeout=cfg.get("timeout", 15),
        )