"""Terminal demo of the multi-agent research assistant (streams events as they happen).

Run from google_adk_application/:
    python multi_agent.py            # scripted demo conversation
    python multi_agent.py --chat     # interactive chat

For the HTTP API with streaming see api.py.
"""

import asyncio
import logging
import sys
import warnings

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import database
from agents import root_agent
from config import APP_NAME, RESEARCH_MODEL

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

USER_ID = "user_1"
SESSION_ID = "session_001"


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str) -> None:
    """Send one message and print the streamed tokens, tool calls and handoffs."""
    print(f"\n>>> User: {query}")
    content = types.Content(role="user", parts=[types.Part(text=query)])
    streamed, current = set(), None

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        for call in event.get_function_calls():
            print(f"\n  [{event.author}] -> tool {call.name}({dict(call.args or {})})")
        for resp in event.get_function_responses():
            print(f"  [{event.author}] <- {resp.name}: {str(resp.response)[:150]}...")
        if event.actions and event.actions.transfer_to_agent:
            print(f"  [handoff] {event.author} -> {event.actions.transfer_to_agent}")

        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text and not p.thought)
            if not text:
                continue
            if event.author != current:
                print(f"\n<<< {event.author}: ", end="")
                current = event.author
            if event.partial:
                streamed.add(event.author)
                print(text, end="", flush=True)
            elif event.author not in streamed:
                print(text, end="", flush=True)
    print()


async def main(interactive: bool) -> None:
    database.init_db()
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"user:name": "Bassant"},  # initial state; prefixes control the scope
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    print(f"Agent '{root_agent.name}' running on model '{RESEARCH_MODEL}'.")

    if interactive:
        while (query := input("\nYou (empty to quit): ").strip()):
            await call_agent_async(query, runner, USER_ID, SESSION_ID)
    else:
        for query in [
            "I'm researching retrieval augmented generation evaluation.",
            "Find 3 recent arXiv papers on evaluating RAG systems.",
            "Save the first paper to my library with the note 'baseline metrics'.",
            "What does my library say about how RAG answers are evaluated?",
        ]:
            await call_agent_async(query, runner, USER_ID, SESSION_ID)

    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    print("\n--- Final session state ---")
    print("topic:", session.state.get("research_topic"))
    print("papers in session:", list(session.state.get("papers", {}).keys()))
    print("saved record ids:", session.state.get("saved_record_ids"))
    print("tool log:", [t["tool"] for t in session.state.get("tool_log", [])])


if __name__ == "__main__":
    asyncio.run(main(interactive="--chat" in sys.argv))
