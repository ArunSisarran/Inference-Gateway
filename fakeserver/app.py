import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

FAKE_REPLY = "Hello! This is a fake response from the fake model server."
WORDS = FAKE_REPLY.split(" ")
TOKEN_DELAY_SECONDS = 0.05  # pause between words, like a real model generating


def make_chunk(completion_id, model, content=None, finish_reason=None):
    """Build one streaming chunk in the OpenAI format."""
    delta = {}
    if content is not None:
        delta["content"] = content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def stream_tokens(model):
    """Yield the reply one word at a time as server-sent events."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    for i, word in enumerate(WORDS):
        await asyncio.sleep(TOKEN_DELAY_SECONDS)
        text = word if i == 0 else " " + word
        chunk = make_chunk(completion_id, model, content=text)
        yield f"data: {json.dumps(chunk)}\n\n"
    final = make_chunk(completion_id, model, finish_reason="stop")
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "fake-model")

    if body.get("stream", False):
        return StreamingResponse(stream_tokens(model), media_type="text/event-stream")

    # Non-streaming: wait for the whole reply, then return it at once.
    await asyncio.sleep(TOKEN_DELAY_SECONDS * len(WORDS))
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": FAKE_REPLY},
                    "finish_reason": "stop",
                }
            ],
        }
    )
