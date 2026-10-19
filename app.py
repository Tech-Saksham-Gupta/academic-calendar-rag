"""Web server for the Academic Calendar Assistant."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from rag import ai_status, answer_question, get_documents, index_pdf, remove_document, suggest_follow_up_questions

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
logger = logging.getLogger(__name__)

app = FastAPI(title="Academic Calendar Assistant")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)


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
    original_name = file.filename or ""
    try:
        if Path(original_name.replace("\\", "/")).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        contents = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="PDF is larger than the 20 MB upload limit.")
        if b"%PDF" not in contents[:1_024]:
            raise HTTPException(status_code=400, detail="This file does not look like a valid PDF.")
        safe_name = Path(original_name.replace("\\", "/")).name[:180]
        return index_pdf(safe_name, contents)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Calendar PDF upload failed")
        raise HTTPException(status_code=422, detail="Could not read this PDF. Please try a text-based, unlocked PDF.") from error
    finally:
        await file.close()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    if not remove_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True}


@app.post("/api/chat")
def chat(payload: Question):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    try:
        result = answer_question(question)
        result["suggestions"] = suggest_follow_up_questions(question)
        return result
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Calendar question failed")
        raise HTTPException(status_code=502, detail="I couldn’t answer that just now. Please try again.") from error
