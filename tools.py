"""Tools for the research assistant agents.

ADK turns any Python function into a tool: the function name, type hints and
docstring become the schema the LLM sees. A parameter named `tool_context`
is injected by ADK (hidden from the LLM) and gives access to session state,
the user id, etc.

LangChain tools (here: Tavily web search) are plugged in with ADK's
`LangchainTool` wrapper; LangChain's ArxivRetriever powers arxiv_search.

  set_research_topic  -> state: research_topic, user:research_interests
  arxiv_search        -> LangChain ArxivRetriever, records papers in state["papers"]
  web_search          -> LangChain TavilySearch wrapped with LangchainTool
  search_database     -> keyword search in the SQLite research library
  save_to_database    -> insert a record (and index it for RAG)
  save_paper          -> save a paper found earlier in this session (from state)
  rag_search          -> semantic retrieval over the saved library
  get_research_state  -> summary of what the session has gathered so far
"""

import os
import re

from google.adk.integrations.langchain import LangchainTool
from google.adk.tools import ToolContext
from langchain_community.retrievers import ArxivRetriever

import database

# ---------------------------------------------------------------------------
# State keys (one place, so agents/callbacks/tools agree)
#   no prefix -> this session only
#   user:     -> shared by all sessions of the same user (needs a persistent session service)
#   app:      -> shared by every user of the app
#   temp:     -> this invocation only, never persisted
# ---------------------------------------------------------------------------
TOPIC = "research_topic"
PAPERS = "papers"                  # dict[arxiv_id, paper dict]
WEB_SOURCES = "web_sources"        # list[{title, url}]
SAVED_IDS = "saved_record_ids"     # list[int]
USER_INTERESTS = "user:research_interests"
USER_SAVED_COUNT = "user:saved_count"


def _arxiv_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/2405.01234v2' -> '2405.01234'"""
    return re.sub(r"v\d+$", "", entry_id.rstrip("/").split("/")[-1])


# ---------------------------------------------------------------------------
# Session-state tools
# ---------------------------------------------------------------------------
def set_research_topic(topic: str, tool_context: ToolContext) -> dict:
    """Set the current research topic for this session. Call this when the user
    states what they are researching, before searching for papers.

    Args:
        topic: Short description of the research topic, e.g. "retrieval augmented generation for code".
    """
    tool_context.state[TOPIC] = topic
    # State changes are only tracked on assignment, so build a new list rather
    # than appending to the stored one in place.
    interests = list(tool_context.state.get(USER_INTERESTS, []))
    if topic not in interests:
        interests = (interests + [topic])[-10:]
        tool_context.state[USER_INTERESTS] = interests
    return {"status": "success", "research_topic": topic, "user_interests": interests}


def get_research_state(tool_context: ToolContext) -> dict:
    """Return a summary of the current research session: topic, papers found so
    far (with their arxiv ids), web sources and ids of records saved."""
    papers = tool_context.state.get(PAPERS, {})
    return {
        "research_topic": tool_context.state.get(TOPIC),
        "papers_found": [
            {"arxiv_id": pid, "title": p["title"], "published": p["published"]}
            for pid, p in papers.items()
        ],
        "web_sources": tool_context.state.get(WEB_SOURCES, []),
        "saved_record_ids": tool_context.state.get(SAVED_IDS, []),
        "user_interests": tool_context.state.get(USER_INTERESTS, []),
    }


# ---------------------------------------------------------------------------
# arXiv search (LangChain ArxivRetriever)
# ---------------------------------------------------------------------------
def arxiv_search(query: str, tool_context: ToolContext, max_results: int = 5) -> dict:
    """Search arXiv for scientific papers. Use for academic / scientific questions.

    Args:
        query: Search query, e.g. "mixture of experts routing".
        max_results: Number of papers to return (1-10).
    """
    max_results = max(1, min(int(max_results), 10))
    try:
        docs = ArxivRetriever(load_max_docs=max_results, doc_content_chars_max=1500).invoke(query)
    except Exception as e:  # network / API errors -> let the agent recover
        return {"status": "error", "error_message": f"arXiv search failed: {e}"}

    results = []
    for d in docs:
        pid = _arxiv_id(d.metadata.get("Entry ID", ""))
        results.append(
            {
                "arxiv_id": pid,
                "title": d.metadata.get("Title", ""),
                "authors": d.metadata.get("Authors", ""),
                "published": str(d.metadata.get("Published", "")),
                "url": f"https://arxiv.org/abs/{pid}",
                "summary": d.page_content,
            }
        )

    # Remember every paper in session state so other agents/tools can use it.
    papers = dict(tool_context.state.get(PAPERS, {}))
    papers.update({p["arxiv_id"]: p for p in results})
    tool_context.state[PAPERS] = papers

    if not results:
        return {"status": "success", "papers": [], "message": f"No arXiv papers for '{query}'."}
    return {"status": "success", "papers": results}


# ---------------------------------------------------------------------------
# Web search (LangChain TavilySearch wrapped as an ADK tool)
# ---------------------------------------------------------------------------
def _web_search_unavailable(query: str) -> dict:
    """Search the public web for current information, blog posts, docs and news.

    Args:
        query: What to search for.
    """
    return {"status": "error", "error_message": "Web search is not configured (set TAVILY_API_KEY)."}


if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch

    web_search = LangchainTool(
        tool=TavilySearch(max_results=5, topic="general"),
        name="web_search",
        description="Search the public web for current information, blog posts, "
        "documentation and news. Input: a search query.",
    )
else:
    web_search = _web_search_unavailable
    web_search.__name__ = "web_search"


# ---------------------------------------------------------------------------
# Database tools (SQLite research library)
# ---------------------------------------------------------------------------
def search_database(
    query: str, tool_context: ToolContext, kind: str = "", limit: int = 5
) -> dict:
    """Keyword search in the user's saved research library (papers, notes, web
    sources, reports). Use when the user asks what they have already saved.

    Args:
        query: Keywords to look for in title, content, authors or tags.
        kind: Optional filter: "paper", "note", "web" or "report". Empty for all.
        limit: Max number of records (1-20).
    """
    rows = database.search_records(
        tool_context.user_id, query, kind or None, max(1, min(int(limit), 20))
    )
    for r in rows:
        r["content"] = r["content"][:500]  # keep the LLM context small
    return {"status": "success", "count": len(rows), "records": rows}


def save_to_database(
    title: str,
    content: str,
    tool_context: ToolContext,
    kind: str = "note",
    source_url: str = "",
    authors: str = "",
    tags: str = "",
) -> dict:
    """Save a record into the user's research library. It is also indexed for
    rag_search. Only use when the user asks to save/store/remember something.

    Args:
        title: Short, unique title for the record.
        content: Full text to store (summary, notes, report...).
        kind: "paper", "note", "web" or "report".
        source_url: Optional link to the source.
        authors: Optional comma separated authors.
        tags: Optional comma separated tags, e.g. "rag,llm".
    """
    try:
        record_id, created = database.save_record(
            user_id=tool_context.user_id,
            kind=kind,
            title=title,
            content=content,
            source_url=source_url or None,
            authors=authors or None,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
        )
    except Exception as e:
        return {"status": "error", "error_message": f"Save failed: {e}"}

    if not created:
        return {"status": "success", "record_id": record_id, "message": "Already saved."}

    tool_context.state[SAVED_IDS] = list(tool_context.state.get(SAVED_IDS, [])) + [record_id]
    tool_context.state[USER_SAVED_COUNT] = tool_context.state.get(USER_SAVED_COUNT, 0) + 1
    return {"status": "success", "record_id": record_id, "message": f"Saved '{title}'."}


def save_paper(arxiv_id: str, tool_context: ToolContext, note: str = "") -> dict:
    """Save an arXiv paper that was found earlier in this session (by arxiv_search)
    into the research library, using the metadata kept in session state.

    Args:
        arxiv_id: The arXiv id, e.g. "2405.01234".
        note: Optional personal note to store with the paper.
    """
    paper = tool_context.state.get(PAPERS, {}).get(arxiv_id)
    if not paper:
        return {
            "status": "error",
            "error_message": f"Paper {arxiv_id} was not found in this session. "
            "Run arxiv_search first or call get_research_state to list known papers.",
        }
    content = paper["summary"] + (f"\n\nNote: {note}" if note else "")
    return save_to_database(
        title=paper["title"],
        content=content,
        tool_context=tool_context,
        kind="paper",
        source_url=paper["url"],
        authors=paper["authors"],
        tags=tool_context.state.get(TOPIC, "") or "",
    )


def rag_search(question: str, tool_context: ToolContext, k: int = 4) -> dict:
    """Answer questions from the user's saved research library using semantic
    (embedding) search. Returns the most relevant passages with their source.
    Prefer this over the web when the user asks about papers/notes they saved.

    Args:
        question: Natural language question.
        k: Number of passages to return (1-10).
    """
    try:
        passages = database.rag_retrieve(tool_context.user_id, question, max(1, min(int(k), 10)))
    except Exception as e:
        return {"status": "error", "error_message": f"RAG retrieval failed: {e}"}
    if not passages:
        return {"status": "success", "passages": [], "message": "The library is empty or has no match."}
    return {"status": "success", "passages": passages}
