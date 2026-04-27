import logging
from typing import AsyncIterator

import websockets

from inference.core.types import AudioChunk, PluginConfig
from inference.plugins.tts.base import TTSPlugin

SENTENCE_ENDERS = {"。", "！", "？", ".", "!", "?", "；", ";"}


class CartesiaTTSPlugin(TTSPlugin):
    name = "tts.cartesia"

    def __init__(self) -> None:
        self.api_key = ""
        self.voice_id = ""
        self.ws_url = "wss://api.cartesia.ai/tts/websocket"
        self.model_id = "sonic-3"
        self._chunk_size = 8000
        self._buffer = b""

    async def initialize(self, config: PluginConfig) -> None:
        self.api_key = config.params.get("api_key", "")
        self.voice_id = config.params.get("voice_id", "")
        self.ws_url = config.params.get(
            "ws_url", "wss://api.cartesia.ai/tts/websocket"
        )
        self.model_id = config.params.get("model_id", "sonic-3")

    async def synthesize_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[AudioChunk]:
        import asyncio
        import base64
        import json
        import uuid

        full_text = ""
        async for sentence in text_stream:
            if not sentence.strip():
                continue
            full_text += sentence + " "

        if not full_text.strip():
            return

        context_id = str(uuid.uuid4())
        try:
            async with websockets.connect(
                self.ws_url,
                extra_headers={"X-API-Key": self.api_key},
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "model_id": self.model_id,
                            "transcript": full_text.strip(),
                            "voice": {"mode": "id", "id": self.voice_id},
                            "output_format": {
                                "container": "raw",
                                "encoding": "pcm_s16le",
                                "sample_rate": 16000,
                            },
                            "context_id": context_id,
                            "continue": False,
                        }
                    )
                )

                async for message in ws:
                    if isinstance(message, bytes):
                        message = message.decode("utf-8")

                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")
                    if msg_type == "error":
                        logging.error("Cartesia error: %s", data.get("message"))
                        continue

                    if msg_type == "chunk":
                        audio_data = data.get("data", "")
                        if audio_data:
                            audio_bytes = base64.b64decode(audio_data)
                            self._buffer += audio_bytes
                            while len(self._buffer) >= self._chunk_size * 2:
                                chunk_data = self._buffer[: self._chunk_size * 2]
                                self._buffer = self._buffer[self._chunk_size * 2 :]
                                yield AudioChunk(
                                    data=chunk_data,
                                    sample_rate=16000,
                                    channels=1,
                                    format="pcm_s16le",
                                    is_final=False,
                                )

                    elif msg_type == "done":
                        break

        except Exception:
            logging.exception("Cartesia TTS error")
            return

        if self._buffer:
            yield AudioChunk(
                data=self._buffer,
                sample_rate=16000,
                channels=1,
                format="pcm_s16le",
                is_final=True,
            )
            self._buffer = b""

    async def shutdown(self) -> None:
        pass