# Research Assistant (Google ADK)

A multi-agent research assistant built with [Google ADK](https://google.github.io/adk-docs/).
It searches arXiv and the web, writes literature reviews, and keeps a personal
research library (SQLite + RAG) that you can save to and query.

## Agents

```
research_coordinator (root, routes requests)
├── knowledge_librarian      save papers/notes, answer from your library (RAG)
└── deep_research_pipeline   SequentialAgent
    ├── source_researcher    arxiv_search, web_search
    └── report_writer        writes the final report
```

Agents share data through session state.

## Project layout

| File             | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| `agents.py`      | Agent definitions (`root_agent`)                           |
| `tools.py`       | Tools: arXiv/web search, save to library, RAG search       |
| `callbacks.py`   | Model/tool callbacks: guards, caching, state tracking      |
| `database.py`    | SQLite research library + embeddings for RAG               |
| `config.py`      | Loads `.env`, chooses the model and embedding provider     |
| `multi_agent.py` | Terminal demo / interactive chat                           |
| `api.py`         | FastAPI server with SSE streaming                          |

## Setup

```bash
pip install google-adk litellm fastapi uvicorn python-dotenv numpy \
    langchain-core langchain-community langchain-text-splitters \
    langchain-tavily arxiv
# optional: langchain-openai, langchain-ollama
```

Create a `.env` file in this folder:

```env
GOOGLE_API_KEY=...
TAVILY_API_KEY=...                    # needed for web_search
RESEARCH_MODEL=gemini-flash-latest    # or openai/gpt-4o-mini, ollama_chat/qwen2.5:7b
EMBEDDING_PROVIDER=google             # or openai / ollama
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
# OLLAMA_API_BASE=http://127.0.0.1:11434
```

Gemini models are used natively; any other model goes through LiteLLM.
Small local models often fail to delegate between agents. `qwen2.5:7b` is the
smallest one worth trying.

## Run

From this folder:

```bash
python multi_agent.py           # scripted demo conversation
python multi_agent.py --chat    # interactive chat in the terminal
```

Or start the API:

```bash
uvicorn api:app --reload --port 8080
```

```bash
curl -X POST localhost:8080/sessions -H 'Content-Type: application/json' -d '{"user_id":"u1"}'
curl -N -X POST localhost:8080/chat/stream -H 'Content-Type: application/json' \
     -d '{"user_id":"u1","session_id":"<id>","message":"Find recent papers on RAG evaluation"}'
```

### API endpoints

| Method | Path                                    | Description                       |
| ------ | --------------------------------------- | --------------------------------- |
| POST   | `/sessions`                             | Create a session                  |
| GET    | `/sessions/{user_id}`                   | List a user's sessions            |
| GET    | `/sessions/{user_id}/{session_id}/state`| Get session state                 |
| POST   | `/chat`                                 | Send a message, get the full reply|
| POST   | `/chat/stream`                          | Send a message, stream events (SSE)|
| POST   | `/library/ingest`                       | Add a document to the library     |
| GET    | `/library/{user_id}/search?q=...`       | Search the library                |

## Data

- `research_library.db`: saved papers, notes, and reports, with their embeddings
- `sessions.db`: API chat sessions

Both are git-ignored, as is `.env`.
