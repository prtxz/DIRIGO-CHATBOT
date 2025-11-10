# app/main.py
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from .model_api import generate_response, generate_response_stream, load_model

app = FastAPI(title="Dirigo Chat Backend")

# Allow all origins for local dev; in production restrict this to your frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Ensure model/data are loaded once at startup (optional)
    try:
        load_model()
    except Exception as e:
        # print to logs — the server will still start so you can see errors in terminal
        print("Warning: load_model() raised on startup:", e)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

def event_stream(prompt: str):
    """
    Yield Server-Sent Events (SSE) containing JSON payloads.
    Each yielded line is a single SSE message (data: {json}\n\n).
    """
    for chunk in generate_response_stream(prompt):
        payload = json.dumps({"type": "message_chunk", "chunk": chunk})
        yield f"data: {payload}\n\n"
    # final event
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.get("/chat/stream")
async def chat_stream(q: str = Query(..., description="User prompt")):
    """
    SSE endpoint. Browser can connect with EventSource to stream replies.
    Example: /chat/stream?q=hello
    """
    return StreamingResponse(event_stream(q), media_type="text/event-stream")

@app.post("/chat")
async def chat_post(body: dict):
    """
    Synchronous endpoint (returns full response at once).
    Useful for quick tests from the frontend.
    """
    prompt = body.get("message", "")
    resp = generate_response(prompt)
    return {"response": resp}
