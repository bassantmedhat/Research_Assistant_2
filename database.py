"""Research library: a SQLite database of saved papers/notes + a vector index for RAG.

Tables
  records : one row per saved item (paper, note, web source, report)
  chunks  : text chunks of each record with their embedding (JSON list of floats)

RAG = embed the question, cosine-similarity against `chunks`, return the top-k
chunks together with the record they came from. LangChain provides the text
splitter and the embeddings interface.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    EMBEDDING_PROVIDER,
    GOOGLE_EMBEDDING_MODEL,
    LIBRARY_DB_PATH,
    OLLAMA_EMBEDDING_MODEL,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- paper | note | web | report
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    source_url  TEXT,
    authors     TEXT,
    tags        TEXT,                       -- JSON list
    created_at  TEXT NOT NULL,
    UNIQUE(user_id, title)
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL               -- JSON list[float]
);
CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id);
"""

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)


# ---------------------------------------------------------------------------
# Embeddings (LangChain interface)
# ---------------------------------------------------------------------------
class GeminiEmbeddings(Embeddings):
    """Minimal LangChain Embeddings wrapper around the google-genai SDK."""

    def __init__(self, model: str = GOOGLE_EMBEDDING_MODEL):
        from google import genai

        self.client = genai.Client()
        self.model = model

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in result.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    return GeminiEmbeddings()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
@contextmanager
def _connect():
    conn = sqlite3.connect(LIBRARY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


# ---------------------------------------------------------------------------
# Public API used by the tools
# ---------------------------------------------------------------------------
def save_record(
    user_id: str,
    kind: str,
    title: str,
    content: str,
    source_url: Optional[str] = None,
    authors: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> tuple[int, bool]:
    """Insert a record and index its chunks for RAG.

    Returns (record_id, created). If the user already saved a record with the
    same title, nothing is written and created=False.
    """
    init_db()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM records WHERE user_id = ? AND title = ?", (user_id, title)
        ).fetchone()
        if existing:
            return existing["id"], False

        # Embed before inserting so a failing embedding call leaves no orphan row.
        chunks = _splitter.split_text(f"{title}\n\n{content}")
        vectors = get_embeddings().embed_documents(chunks)

        cur = conn.execute(
            """INSERT INTO records (user_id, kind, title, content, source_url, authors, tags, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                kind,
                title,
                content,
                source_url,
                authors,
                json.dumps(tags or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        record_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO chunks (record_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
            [(record_id, i, c, json.dumps(v)) for i, (c, v) in enumerate(zip(chunks, vectors))],
        )
        return record_id, True


def search_records(
    user_id: str, query: str, kind: Optional[str] = None, limit: int = 5
) -> list[dict]:
    """Keyword search over title/content/authors/tags (every word must match)."""
    init_db()
    sql = "SELECT * FROM records WHERE user_id = ?"
    params: list = [user_id]
    for word in query.split():
        sql += " AND (title LIKE ? OR content LIKE ? OR authors LIKE ? OR tags LIKE ?)"
        params += [f"%{word}%"] * 4
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def rag_retrieve(user_id: str, question: str, k: int = 4) -> list[dict]:
    """Semantic retrieval: top-k chunks of the user's library by cosine similarity."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT c.text, c.embedding, r.id AS record_id, r.title, r.source_url, r.kind
               FROM chunks c JOIN records r ON r.id = c.record_id
               WHERE r.user_id = ?""",
            (user_id,),
        ).fetchall()
    if not rows:
        return []

    matrix = np.array([json.loads(r["embedding"]) for r in rows], dtype=np.float32)
    q = np.array(get_embeddings().embed_query(question), dtype=np.float32)
    scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10)
    top = np.argsort(-scores)[:k]
    return [
        {
            "record_id": rows[i]["record_id"],
            "title": rows[i]["title"],
            "kind": rows[i]["kind"],
            "source_url": rows[i]["source_url"],
            "score": round(float(scores[i]), 4),
            "text": rows[i]["text"],
        }
        for i in top
    ]
