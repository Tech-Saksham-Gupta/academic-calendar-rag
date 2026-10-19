"""Web server for the Academic Calendar Assistant."""

import base64
import hmac
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from rag import ai_status, answer_question, get_documents, index_pdf, remove_document

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="Academic Calendar Assistant")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def protect_deployed_site(request: Request, call_next):
    """Ask for the Render login when deployment credentials are configured."""
    username = os.getenv("APP_USERNAME", "")
    password = os.getenv("APP_PASSWORD", "")
    if not username or not password or request.url.path == "/api/health":
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Basic "):
        try:
            decoded = base64.b64decode(authorization[6:]).decode("utf-8")
            provided_username, provided_password = decoded.split(":", 1)
            user_matches = hmac.compare_digest(provided_username, username)
            password_matches = hmac.compare_digest(provided_password, password)
            if user_matches and password_matches:
                return await call_next(request)
        except (ValueError, UnicodeDecodeError):
            pass

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Academic Calendar Assistant"'},
        content="Sign in to use the Academic Calendar Assistant.",
    )


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return ai_status()


@app.get("/api/documents")
def documents():
    return {"documents": get_documents()}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="PDF is larger than the 20 MB upload limit.")
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="This file does not look like a valid PDF.")

    try:
        return index_pdf(file.filename, contents)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not process the PDF: {error}") from error


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    if not remove_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True}


@app.post("/api/chat")
def chat(payload: Question):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    try:
        return answer_question(payload.question.strip())
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"The assistant could not answer: {error}") from error
