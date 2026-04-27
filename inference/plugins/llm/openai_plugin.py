from typing import AsyncIterator

from inference.core.types import LLMResponseChunk, PluginConfig
from inference.plugins.llm.base import LLMPlugin

SENTENCE_ENDERS = {"。", "！", "？", ".", "!", "?", "；", ";", "\n"}


class OpenRouterLLMPlugin(LLMPlugin):
    name = "llm.openrouter"

    def __init__(self) -> None:
        self.client = None
        self.model = "google/gemini-2.0-flash-001"
        self.temperature = 0.7
        self.system_prompt = ""

    async def initialize(self, config: PluginConfig) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=config.params.get("api_key"),
            base_url=config.params.get("base_url", "https://openrouter.ai/api/v1"),
        )
        self.model = config.params.get("model", "google/gemini-2.0-flash-001")
        self.temperature = float(config.params.get("temperature", 0.7))
        self.system_prompt = config.params.get("system_prompt", "")

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[LLMResponseChunk]:
        full_messages = messages
        if self.system_prompt:
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        accumulated = ""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                accumulated += token
                is_sentence_end = any(token.endswith(p) for p in SENTENCE_ENDERS)
                yield LLMResponseChunk(
                    token=token,
                    accumulated_text=accumulated,
                    is_sentence_end=is_sentence_end,
                    is_final=False,
                )

        yield LLMResponseChunk(
            token="",
            accumulated_text=accumulated,
            is_sentence_end=True,
            is_final=True,
        )

    async def shutdown(self) -> None:
        self.client = None