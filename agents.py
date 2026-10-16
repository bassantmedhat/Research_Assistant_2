"""Multi-agent research assistant built with Google ADK.

research_coordinator (root, routes requests with transfer_to_agent)
├── knowledge_librarian     rag_search, search_database,      "what did I save about X?"
│                           save_paper, save_to_database      "save that paper / note"
└── deep_research_pipeline  SequentialAgent                   "write me a literature review on X"
    ├── source_researcher   arxiv_search, web_search  -> state["research_findings"]
    └── report_writer                                 -> state["research_report"]

Agents share data through session state: tools write to it, `output_key`
stores an agent's final answer in it, and `{key?}` in an instruction reads
it (the `?` makes the key optional).
"""

from google.adk.agents import LlmAgent, SequentialAgent

from callbacks import (
    guard_model_calls,
    init_state,
    log_turn_end,
    track_tool_results,
    validate_and_cache_tools,
)
from config import get_model
from tools import (
    arxiv_search,
    get_research_state,
    rag_search,
    save_paper,
    save_to_database,
    search_database,
    set_research_topic,
    web_search,
)

MODEL = get_model()

# Callbacks shared by every LLM agent that uses tools.
TOOL_AGENT_CALLBACKS = dict(
    before_model_callback=guard_model_calls,
    before_tool_callback=validate_and_cache_tools,
    after_tool_callback=track_tool_results,
)

BACK_TO_COORDINATOR = (
    "If the user asks for something outside your job, transfer back to "
    "research_coordinator."
)

# ---------------------------------------------------------------------------
# Specialist sub-agents
# ---------------------------------------------------------------------------
knowledge_librarian = LlmAgent(
    name="knowledge_librarian",
    model=MODEL,
    description="Manages the user's research library: saves papers, notes and reports, "
    "and answers questions from what is saved (database + RAG).",
    instruction=(
        "You manage the user's saved research library.\n"
        "Saving:\n"
        "- For a paper found earlier in this session, use save_paper with its arXiv id. "
        "Use get_research_state if you need to look up ids.\n"
        "- For notes, web sources or reports, use save_to_database with a clear title.\n"
        "- If the user says 'save the report', the latest report is:\n{research_report?}\n"
        "- Confirm what was saved with its record id.\n"
        "Answering:\n"
        "- For questions about content ('what does my library say about X'), use rag_search.\n"
        "- For listing/finding saved items by keyword, author or type, use search_database.\n"
        "- Answer from the saved library only and cite the record title for every claim. "
        "If nothing relevant is saved, say so and suggest a literature search.\n"
        + BACK_TO_COORDINATOR
    ),
    tools=[rag_search, search_database, save_paper, save_to_database, get_research_state],
    **TOOL_AGENT_CALLBACKS,
)

# ---------------------------------------------------------------------------
# Deep research workflow (deterministic: gather -> write)
# ---------------------------------------------------------------------------
source_researcher = LlmAgent(
    name="source_researcher",
    model=MODEL,
    description="Collects academic papers and recent web sources for the deep research pipeline.",
    instruction=(
        "Research topic: {research_topic?}. Use the user's latest request if the topic is not set.\n"
        "- Run 2-3 focused arxiv_search queries covering different angles of the topic.\n"
        "- Run 1-2 web_search queries for recent developments, tools and benchmarks.\n"
        "Output two bullet lists:\n"
        "Academic findings: title, year, arXiv id, key contribution.\n"
        "Web findings: source title, URL, key point."
    ),
    tools=[arxiv_search, web_search],
    output_key="research_findings",
    **TOOL_AGENT_CALLBACKS,
)

report_writer = LlmAgent(
    name="report_writer",
    model=MODEL,
    description="Writes the final literature review from the gathered findings.",
    instruction=(
        "Write a concise literature review on: {research_topic?}\n\n"
        "Findings:\n{research_findings?}\n\n"
        "Structure: 1) Overview 2) Key approaches (cite papers by arXiv id) "
        "3) Recent developments 4) Open problems 5) References. "
        "Use only the findings above. End by offering to save the report to the library."
    ),
    output_key="research_report",
    before_model_callback=guard_model_calls,
)

deep_research_pipeline = SequentialAgent(
    name="deep_research_pipeline",
    description="Runs a full literature review: searches arXiv and the web, "
    "then writes a structured report. Use for 'literature review', 'survey', 'deep research'.",
    sub_agents=[
        source_researcher,
        report_writer,
    ],
)

# ---------------------------------------------------------------------------
# Root agent
# ---------------------------------------------------------------------------
root_agent = LlmAgent(
    name="research_coordinator",
    model=MODEL,
    description="Coordinates a team of research agents.",
    instruction=(
        "You are the coordinator of a research assistant team. You do not search yourself; "
        "you route each request to the right specialist with transfer_to_agent.\n\n"
        "Session context:\n"
        "- Research topic: {research_topic?}\n"
        "- User's past interests: {user:research_interests?}\n"
        "- Records saved by this user: {user:saved_count?}\n\n"
        "Routing:\n"
        "- When the user states or changes their topic, call set_research_topic first.\n"
        "- Save / store / remember requests, or questions about their saved library "
        "-> knowledge_librarian\n"
        "- Finding papers or web sources, literature review, survey, in-depth report "
        "-> deep_research_pipeline "
        "(call set_research_topic before transferring)\n"
        "- Greetings or questions about what you can do: answer directly.\n"
        "Use get_research_state if the user asks what has been found so far."
    ),
    tools=[set_research_topic, get_research_state],
    sub_agents=[knowledge_librarian, deep_research_pipeline],
    before_agent_callback=init_state,
    after_agent_callback=log_turn_end,
    **TOOL_AGENT_CALLBACKS,
)
