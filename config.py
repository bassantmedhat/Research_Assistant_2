"""Central configuration: loads secrets from .env and picks the models.

Never hardcode API keys in source files. Put them in google_adk_application/.env:

    GOOGLE_API_KEY=...
    OPENAI_API_KEY=...          # optional
    ANTHROPIC_API_KEY=...       # optional
    TAVILY_API_KEY=...          # needed for web_search
    OLLAMA_API_BASE=http://127.0.0.1:11434
    RESEARCH_MODEL=gemini-flash-latest      # or openai/gpt-4o-mini, ollama_chat/qwen2.5:7b ...
    EMBEDDING_PROVIDER=google               # or openai / ollama
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Use API keys directly (not Vertex AI / Agent Platform).
os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "False")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")
os.environ.setdefault("OLLAMA_API_BASE", "http://127.0.0.1:11434")

APP_NAME = "research_assistant"

# --- Models -----------------------------------------------------------------
# Multi-agent delegation (transfer_to_agent) needs a model that is good at tool
# calling. Gemini / GPT-4o-mini / Claude work well; small local models (3B)
# often fail to delegate. qwen2.5:7b is the smallest local model worth trying.
MODEL_GEMINI_FLASH = "gemini-flash-latest"
MODEL_GPT_4O_MINI = "openai/gpt-4o-mini"
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-6"
MODEL_OLLAMA_QWEN_7B = "ollama_chat/qwen2.5:7b"

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", MODEL_GEMINI_FLASH)

# --- Storage ----------------------------------------------------------------
LIBRARY_DB_PATH = Path(os.getenv("LIBRARY_DB_PATH", BASE_DIR / "research_library.db"))
SESSION_DB_URL = os.getenv(
    "SESSION_DB_URL", f"sqlite+aiosqlite:///{BASE_DIR / 'sessions.db'}"
)

# --- Embeddings (for RAG) ---------------------------------------------------
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER", "google" if os.getenv("GOOGLE_API_KEY") else "ollama"
)
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def get_model(model_name: str = RESEARCH_MODEL):
    """Gemini model ids are used natively by ADK; everything else goes via LiteLLM."""
    if model_name.startswith("gemini"):
        return model_name
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model_name)
