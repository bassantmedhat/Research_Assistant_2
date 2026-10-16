"""FastAPI server with Server-Sent Events streaming for the research assistant.

Run (from google_adk_application/):
    uvicorn api:app --reload --port 8080

Try it:
    curl -X POST localhost:8080/sessions -H 'Content-Type: application/json' -d '{"user_id":"u1"}'
    curl -N -X POST localhost:8080/chat/stream -H 'Content-Type: application/json' \
         -d '{"user_id":"u1","session_id":"<id>","message":"Find recent papers on RAG evaluation"}'

Stream format: one `data: {json}` line per event, where json["type"] is one of
    token        partial text from an agent (append to the UI)
    message      full text of a non-streamed agent message
    tool_call    an agent called a tool
    tool_result  a tool returned
    handoff      control was transferred to another agent
    state        state keys changed by this event
    error        something failed
    done         end of the turn, includes the final session state
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types
from pydantic import BaseModel

import database
from agents import root_agent
from config import APP_NAME, SESSION_DB_URL

logging.basicConfig(level=logging.INFO)

# Sessions (and user:/app: state) survive restarts thanks to the DB session service.
session_service = DatabaseSessionService(db_url=SESSION_DB_URL)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_db()
    yield


app = FastAPI(title="Research Assistant (Google ADK)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    research_topic: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class IngestRequest(BaseModel):
    user_id: str
    title: str
    content: str
    kind: str = "note"
    source_url: Optional[str] = None
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Event -> JSON
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def event_to_messages(event: Event, streamed_authors: set[str]) -> list[dict]:
    """Translate one ADK event into zero or more client messages.

    With StreamingMode.SSE an agent's text arrives as many `partial` events and
    then once more as a complete (non-partial) event. We forward the partials
    as tokens and drop the complete copy, unless that agent never streamed
    (e.g. a model/provider without streaming), in which case we send it whole.
    """
    out: list[dict] = []
    author = event.author

    if event.content and event.content.parts:
        text = "".join(p.text for p in event.content.parts if p.text and not p.thought)
        if text:
            if event.partial:
                streamed_authors.add(author)
                out.append({"type": "token", "agent": author, "text": text})
            elif author not in streamed_authors and author != "user":
                out.append({"type": "message", "agent": author, "text": text})

    for call in event.get_function_calls():
        out.append({"type": "tool_call", "agent": author, "tool": call.name,
                    "args": _jsonable(call.args or {})})
    for resp in event.get_function_responses():
        out.append({"type": "tool_result", "agent": author, "tool": resp.name,
                    "result": _jsonable(resp.response)})

    if event.actions:
        if event.actions.transfer_to_agent:
            out.append({"type": "handoff", "from": author, "to": event.actions.transfer_to_agent})
        delta = {k: v for k, v in (event.actions.state_delta or {}).items() if not k.startswith("temp:")}
        if delta:
            out.append({"type": "state", "agent": author, "delta": _jsonable(delta)})

    if event.error_message:
        out.append({"type": "error", "agent": author, "message": event.error_message})
    return out


def sse(message: dict) -> str:
    return f"data: {json.dumps(message, default=str)}\n\n"


async def run_turn_stream(req: ChatRequest) -> AsyncIterator[str]:
    content = types.Content(role="user", parts=[types.Part(text=req.message)])
    streamed_authors: set[str] = set()
    try:
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=req.session_id,
            new_message=content,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            for message in event_to_messages(event, streamed_authors):
                yield sse(message)
    except Exception as e:  # surface errors to the client instead of cutting the stream
        logging.exception("agent run failed")
        yield sse({"type": "error", "message": str(e)})

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=req.user_id, session_id=req.session_id
    )
    yield sse({"type": "done", "state": _jsonable(session.state if session else {})})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    initial_state = {"research_topic": req.research_topic} if req.research_topic else None
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=req.user_id, session_id=req.session_id, state=initial_state
    )
    return {"session_id": session.id, "user_id": session.user_id, "state": session.state}


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    resp = await session_service.list_sessions(app_name=APP_NAME, user_id=user_id)
    return {"sessions": [{"session_id": s.id, "last_update_time": s.last_update_time}
                         for s in resp.sessions]}


@app.get("/sessions/{user_id}/{session_id}/state")
async def get_state(user_id: str, session_id: str):
    session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return _jsonable(session.state)


async def _require_session(user_id: str, session_id: str) -> None:
    if not await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id):
        raise HTTPException(404, "Session not found. Create it with POST /sessions first.")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    await _require_session(req.user_id, req.session_id)
    return StreamingResponse(
        run_turn_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    """Non-streaming variant: returns the final answer and the agents involved."""
    await _require_session(req.user_id, req.session_id)
    content = types.Content(role="user", parts=[types.Part(text=req.message)])
    final_text, agents_involved = "", []
    async for event in runner.run_async(user_id=req.user_id, session_id=req.session_id, new_message=content):
        if event.author not in agents_involved and event.author != "user":
            agents_involved.append(event.author)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text for p in event.content.parts if p.text and not p.thought)
    return {"response": final_text, "agents": agents_involved}


@app.post("/library/ingest")
def ingest(req: IngestRequest):
    """Add your own documents to the library (they become searchable with RAG)."""
    record_id, created = database.save_record(
        user_id=req.user_id, kind=req.kind, title=req.title, content=req.content,
        source_url=req.source_url, tags=req.tags,
    )
    return {"record_id": record_id, "created": created}


@app.get("/library/{user_id}/search")
def library_search(user_id: str, q: str, limit: int = 10):
    return {"records": database.search_records(user_id, q, limit=limit)}
