"""Readable RAG pipeline with either local Ollama or free-tier Gemini models."""

import json
import os
import re
import sqlite3
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from pypdf import PdfReader
import requests

BASE_DIR = Path(__file__).parent
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "calendar.db")))
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
RESULT_COUNT = 5
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def ai_status():
    """Check whether the selected AI provider is configured and reachable."""
    if AI_PROVIDER == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return {"ready": False, "message": "Add a Gemini API key"}
        return {"ready": True, "message": "Gemini AI ready"}

    return ollama_status()


def ollama_status():
    """Check that Ollama is running and the two local models are installed."""
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
        installed = {model["name"] for model in response.json().get("models", [])}
        chat_ready = any(name == chat_model or name.startswith(chat_model + ":") for name in installed)
        embed_ready = any(name == embedding_model or name.startswith(embedding_model + ":") for name in installed)
        if chat_ready and embed_ready:
            return {"ready": True, "message": "Local AI ready"}
        missing = [name for name, ready in ((chat_model, chat_ready), (embedding_model, embed_ready)) if not ready]
        return {"ready": False, "message": "Download models: " + ", ".join(missing)}
    except requests.RequestException:
        return {"ready": False, "message": "Start Ollama to connect"}


def connect_database():
    """Open the local database and create its table on the first run."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS passages (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )"""
    )
    connection.commit()
    return connection


def split_into_chunks(text):
    """Split text into overlapping passages so nearby context is retained."""
    clean_text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(start + CHUNK_SIZE, len(clean_text))
        if end < len(clean_text):
            word_boundary = clean_text.rfind(" ", start, end)
            if word_boundary > start:
                end = word_boundary
        piece = clean_text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def make_embeddings(texts):
    """Turn text into numeric vectors that can be compared by meaning."""
    if AI_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini is selected, but GEMINI_API_KEY is not set.")
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
        requests_body = [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": f"Represent this for academic calendar retrieval: {text}"}]},
                "embedContentConfig": {"outputDimensionality": 768},
            }
            for text in texts
        ]
        response = requests.post(
            f"{GEMINI_API_URL}/{model}:batchEmbedContents",
            headers={"x-goog-api-key": api_key},
            json={"requests": requests_body},
            timeout=180,
        )
        response.raise_for_status()
        return [item["values"] for item in response.json()["embeddings"]]

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), "input": texts},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def index_pdf(filename, pdf_bytes):
    """Read a PDF, create embeddings, and save searchable passages."""
    reader = PdfReader(BytesIO(pdf_bytes))
    passages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        passages.extend((page_number, chunk) for chunk in split_into_chunks(page_text))

    if not passages:
        raise ValueError("No selectable text found. Scanned PDFs need OCR before upload.")

    document_id = str(uuid.uuid4())
    connection = connect_database()
    try:
        # Small batches make large calendar PDFs more reliable to process.
        for batch_start in range(0, len(passages), 64):
            batch = passages[batch_start : batch_start + 64]
            vectors = make_embeddings([text for _, text in batch])
            connection.executemany(
                "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (str(uuid.uuid4()), document_id, filename, page, text, json.dumps(vector))
                    for (page, text), vector in zip(batch, vectors)
                ],
            )
            connection.commit()
    finally:
        connection.close()

    return {"id": document_id, "filename": filename, "passages": len(passages)}


def get_documents():
    """Return a compact list of indexed calendar files."""
    connection = connect_database()
    rows = connection.execute(
        """SELECT document_id AS id, filename, COUNT(*) AS passages
           FROM passages GROUP BY document_id, filename ORDER BY filename"""
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def remove_document(document_id):
    """Remove one calendar and all of its searchable passages."""
    connection = connect_database()
    cursor = connection.execute("DELETE FROM passages WHERE document_id = ?", (document_id,))
    connection.commit()
    connection.close()
    return cursor.rowcount > 0


def retrieve(question):
    """Find the passages whose embeddings are closest to the question."""
    connection = connect_database()
    rows = connection.execute("SELECT filename, page, text, embedding FROM passages").fetchall()
    connection.close()
    if not rows:
        return []

    question_vector = np.array(make_embeddings([question])[0], dtype=np.float32)
    passage_vectors = np.array([json.loads(row["embedding"]) for row in rows], dtype=np.float32)
    scores = passage_vectors @ question_vector / (
        np.linalg.norm(passage_vectors, axis=1) * np.linalg.norm(question_vector) + 1e-10
    )
    best_indexes = np.argsort(scores)[::-1][:RESULT_COUNT]

    return [
        {
            "filename": rows[index]["filename"],
            "page": rows[index]["page"],
            "text": rows[index]["text"],
            "score": float(scores[index]),
        }
        for index in best_indexes
        if scores[index] > 0.15
    ]


def answer_question(question):
    """Use retrieved evidence to write an answer and return its source cards."""
    sources = retrieve(question)
    if not sources:
        return {
            "answer": "I couldn't find a relevant detail in the indexed calendars. Try another phrasing or upload the calendar that contains the information.",
            "sources": [],
        }

    evidence = "\n\n".join(
        f"[Source {number}: {source['filename']}, page {source['page']}] {source['text']}"
        for number, source in enumerate(sources, start=1)
    )
    instruction = (
        "Answer questions about academic calendars using only the supplied sources. "
        "Do not guess or use outside knowledge. If evidence is missing, say so. "
        "Keep the answer concise and cite claims with labels like [Source 1]."
    )

    if AI_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini is selected, but GEMINI_API_KEY is not set.")
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
        response = requests.post(
            f"{GEMINI_API_URL}/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={
                "systemInstruction": {"parts": [{"text": instruction}]},
                "contents": [{"parts": [{"text": f"Question: {question}\n\nSources:\n{evidence}"}]}],
                "generationConfig": {"temperature": 0.1},
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Question: {question}\n\nSources:\n{evidence}"},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "No answer was returned.")
    return {"answer": answer, "sources": sources}
