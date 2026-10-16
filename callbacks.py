"""ADK callbacks for the research assistant.

Signatures (ADK calls them with these keyword names):
  before_agent(callback_context)                      -> Optional[types.Content]  (skip agent)
  after_agent(callback_context)                       -> Optional[types.Content]
  before_model(callback_context, llm_request)         -> Optional[LlmResponse]   (skip LLM call)
  after_model(callback_context, llm_response)         -> Optional[LlmResponse]
  before_tool(tool, args, tool_context)               -> Optional[dict]          (skip tool)
  after_tool(tool, args, tool_context, tool_response) -> Optional[dict]          (replace result)

Returning None means "carry on normally". Returning a value short-circuits.
"""

import logging
import time
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from tools import PAPERS, SAVED_IDS, TOPIC, WEB_SOURCES

logger = logging.getLogger("research_assistant")

MAX_LLM_CALLS_PER_TURN = 30
TOOL_LOG = "tool_log"
ARXIV_CACHE = "arxiv_cache"  # dict[query, list[arxiv_id]]


# ---------------------------------------------------------------------------
# Agent callbacks
# ---------------------------------------------------------------------------
def init_state(callback_context: CallbackContext) -> Optional[types.Content]:
    """before_agent on the root agent: make sure every state key exists so
    instruction templates like {research_topic} never fail, and start per-turn
    counters in temp: state."""
    state = callback_context.state
    defaults = {
        TOPIC: "not set yet",
        PAPERS: {},
        WEB_SOURCES: [],
        SAVED_IDS: [],
        TOOL_LOG: [],
        ARXIV_CACHE: {},
        "user:research_interests": [],
        "user:saved_count": 0,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value

    state["temp:turn_started_at"] = time.time()
    state["temp:llm_calls"] = 0
    state["app:total_turns"] = state.get("app:total_turns", 0) + 1
    return None


def log_turn_end(callback_context: CallbackContext) -> Optional[types.Content]:
    """after_agent on the root agent: log timing and usage for the turn."""
    started = callback_context.state.get("temp:turn_started_at", time.time())
    logger.info(
        "turn done user=%s session=%s llm_calls=%s papers=%s %.1fs",
        callback_context.user_id,
        callback_context.session.id,
        callback_context.state.get("temp:llm_calls"),
        len(callback_context.state.get(PAPERS, {})),
        time.time() - started,
    )
    return None


# ---------------------------------------------------------------------------
# Model callbacks
# ---------------------------------------------------------------------------
def guard_model_calls(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """before_model on every agent: count LLM calls and stop runaway loops."""
    calls = callback_context.state.get("temp:llm_calls", 0) + 1
    callback_context.state["temp:llm_calls"] = calls
    logger.debug("[%s] LLM call #%d", callback_context.agent_name, calls)

    if calls > MAX_LLM_CALLS_PER_TURN:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="I stopped here because this request needed too many "
                                       "steps. Please narrow the question and try again.")],
            )
        )
    return None


# ---------------------------------------------------------------------------
# Tool callbacks
# ---------------------------------------------------------------------------
def validate_and_cache_tools(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]:
    """before_tool:
    - reject empty/low-quality saves before they hit the database
    - serve repeated arXiv queries from session state instead of the API
    """
    if tool.name == "save_to_database":
        if len(args.get("content", "").strip()) < 20:
            return {"status": "error",
                    "error_message": "Content is too short to save. Provide a real summary or note."}
        if args.get("kind", "note") not in {"paper", "note", "web", "report"}:
            args["kind"] = "note"  # args can be fixed in place before the tool runs

    if tool.name == "arxiv_search":
        query = args.get("query", "").strip().lower()
        cached_ids = tool_context.state.get(ARXIV_CACHE, {}).get(query)
        if cached_ids:
            papers = tool_context.state.get(PAPERS, {})
            logger.info("arxiv cache hit: %s", query)
            return {"status": "success", "cached": True,
                    "papers": [papers[i] for i in cached_ids if i in papers]}
    return None


def track_tool_results(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[dict]:
    """after_tool:
    - keep a short audit log of tool calls in state
    - fill the arXiv cache
    - collect web sources from the (LangChain) Tavily tool into state
    """
    status = tool_response.get("status", "success") if isinstance(tool_response, dict) else "success"
    log = list(tool_context.state.get(TOOL_LOG, []))
    log.append({"agent": tool_context.agent_name, "tool": tool.name,
                "args": {k: str(v)[:80] for k, v in args.items()}, "status": status})
    tool_context.state[TOOL_LOG] = log[-25:]

    if tool.name == "arxiv_search" and isinstance(tool_response, dict) and not tool_response.get("cached"):
        cache = dict(tool_context.state.get(ARXIV_CACHE, {}))
        cache[args.get("query", "").strip().lower()] = [p["arxiv_id"] for p in tool_response.get("papers", [])]
        tool_context.state[ARXIV_CACHE] = cache

    if tool.name == "web_search" and isinstance(tool_response, dict):
        # LangchainTool may return Tavily's dict directly or wrapped in {"result": ...}
        payload = tool_response.get("result", tool_response)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if results:
            sources = list(tool_context.state.get(WEB_SOURCES, []))
            known = {s["url"] for s in sources}
            sources += [{"title": r.get("title", ""), "url": r["url"]}
                        for r in results if r.get("url") and r["url"] not in known]
            tool_context.state[WEB_SOURCES] = sources[-50:]
    return None
