"""Offline checks for PDF indexing and calendar question handling."""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

import app
import rag


def run_async_without_loop(coroutine):
    """Run these route tests' immediately-completing fake upload coroutine."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    raise AssertionError("The fake upload unexpectedly required an event loop")


class FakePage:
    def __init__(self, text=""):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, pages, encrypted=False):
        self.pages = pages
        self.is_encrypted = encrypted

    def decrypt(self, password):
        return 0


class FakeUploadFile:
    """Small async upload stub so route checks do not depend on Windows thread pools."""

    def __init__(self, filename, contents):
        self.filename = filename
        self.contents = contents
        self.closed = False

    async def read(self, size=-1):
        return self.contents[:size]

    async def close(self):
        self.closed = True


class RagChecks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / "work")
        self.original_database = rag.DATABASE_PATH
        self.original_provider = rag.AI_PROVIDER
        self.original_api_key = os.environ.get("GEMINI_API_KEY")
        rag.DATABASE_PATH = Path(self.temp_dir.name) / "calendar.db"
        rag.AI_PROVIDER = "gemini"

    def tearDown(self):
        rag.DATABASE_PATH = self.original_database
        rag.AI_PROVIDER = self.original_provider
        if self.original_api_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.original_api_key
        self.temp_dir.cleanup()

    def add_passage(self, text, page=1, filename="calendar.pdf"):
        connection = rag.connect_database()
        connection.execute(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
            (f"id-{page}-{len(text)}", "doc-1", filename, page, text, "[]"),
        )
        connection.commit()
        connection.close()

    def test_empty_and_long_text_are_handled_by_chunker(self):
        self.assertEqual(rag.split_into_chunks("  \n "), [])
        chunks = rag.split_into_chunks("word " * 900)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))
        self.assertLessEqual(max(map(len, chunks)), rag.CHUNK_SIZE)

    def test_upload_indexes_text_without_gemini_embedding_request(self):
        page_text = "Project submission deadline is October 12, 2026."
        fake_reader = FakeReader([FakePage(page_text)])
        with patch.object(rag, "PdfReader", return_value=fake_reader), patch.object(
            rag, "make_embeddings", side_effect=AssertionError("Gemini embeddings must not be used")
        ):
            result = rag.index_pdf("calendar.pdf", b"%PDF-fake")
        self.assertEqual(result["passages"], 1)
        self.assertEqual(rag.get_documents()[0]["filename"], "calendar.pdf")

    def test_scanned_pdf_is_rejected_without_saving_partial_rows(self):
        with patch.object(rag, "PdfReader", return_value=FakeReader([FakePage("")])):
            with self.assertRaisesRegex(ValueError, "No selectable text"):
                rag.index_pdf("scan.pdf", b"%PDF-fake")
        self.assertEqual(rag.get_documents(), [])

    def test_encrypted_pdf_is_rejected(self):
        with patch.object(rag, "PdfReader", return_value=FakeReader([FakePage("text")], encrypted=True)):
            with self.assertRaisesRegex(ValueError, "password-protected"):
                rag.index_pdf("locked.pdf", b"%PDF-fake")

    def test_oversized_page_count_is_rejected(self):
        pages = [FakePage("text") for _ in range(rag.MAX_PDF_PAGES + 1)]
        with patch.object(rag, "PdfReader", return_value=FakeReader(pages)):
            with self.assertRaisesRegex(ValueError, "more than"):
                rag.index_pdf("large.pdf", b"%PDF-fake")

    def test_failed_database_write_does_not_leave_partial_document(self):
        connection = rag.connect_database()
        connection.execute(
            "CREATE TRIGGER reject_passage BEFORE INSERT ON passages "
            "BEGIN SELECT RAISE(ABORT, 'test write failure'); END"
        )
        connection.commit()
        connection.close()
        with patch.object(rag, "PdfReader", return_value=FakeReader([FakePage("A dated deadline October 12, 2026.")])):
            with self.assertRaises(sqlite3.IntegrityError):
                rag.index_pdf("calendar.pdf", b"%PDF-fake")
        connection = rag.connect_database()
        count = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        connection.close()
        self.assertEqual(count, 0)

    def test_plural_deadline_question_retrieves_project_due_date(self):
        self.add_passage("Project submission deadline is October 12, 2026.")
        self.add_passage("Autumn classes begin August 24, 2026.", page=2)
        results = rag.retrieve("What is the deadline to complete this project?")
        self.assertTrue(results)
        self.assertIn("Project submission deadline", results[0]["text"])

    def test_project_answer_is_short_and_has_advice_but_no_source_label(self):
        sources = [{
            "filename": "calendar.pdf",
            "page": 1,
            "text": "Project submission deadline is October 12, 2026.",
        }]
        answer = rag._local_calendar_answer("What is the deadline to complete this project?", sources)
        self.assertIn("October 12, 2026", answer)
        self.assertIn("submit early", answer)
        self.assertNotIn("Source", answer)
        self.assertLessEqual(len([part for part in answer.split(".") if part.strip()]), 2)

    def test_missing_project_date_is_not_invented(self):
        sources = [{
            "filename": "calendar.pdf",
            "page": 1,
            "text": "Add/drop deadline is September 4, 2026.",
        }]
        answer = rag._local_calendar_answer("When is my project due?", sources)
        self.assertIn("couldn’t find a project completion deadline", answer)
        self.assertNotIn("September 4", answer)

    def test_table_extraction_keeps_event_attached_to_date(self):
        source = {
            "filename": "calendar.pdf",
            "page": 1,
            "text": "DATE EVENT August 24, 2026 First day of classes Autumn semester instruction begins. September 4, 2026 Add/drop deadline.",
        }
        answer = rag._local_calendar_answer("When do Autumn classes begin?", [source])
        self.assertIn("August 24, 2026", answer)
        self.assertIn("First day of classes", answer)

    def test_numeric_date_in_deadline_is_recognized(self):
        source = {
            "filename": "calendar.pdf",
            "page": 1,
            "text": "Project submission deadline is 10/12/2026.",
        }
        answer = rag._local_calendar_answer("When is the project deadline?", [source])
        self.assertIn("10/12/2026", answer)

    def test_gemini_rate_limit_returns_calendar_answer_without_error_text(self):
        self.add_passage("Autumn classes begin Monday, August 24, 2026.")
        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(rag.requests, "post", side_effect=rag.requests.HTTPError("429")):
            result = rag.answer_question("When does Autumn semester begin?")
        self.assertIn("August 24, 2026", result["answer"])
        self.assertNotIn("rate limit", result["answer"].lower())

    def test_missing_gemini_key_still_uses_local_rag(self):
        self.add_passage("Project submission deadline is October 12, 2026.")
        os.environ.pop("GEMINI_API_KEY", None)
        result = rag.answer_question("When is the project deadline?")
        self.assertIn("October 12, 2026", result["answer"])

    def test_upload_route_rejects_non_pdf(self):
        upload = FakeUploadFile(filename="calendar.txt", contents=b"not a pdf")
        with self.assertRaises(HTTPException) as error:
            run_async_without_loop(app.upload_document(upload))
        self.assertEqual(error.exception.status_code, 400)
        self.assertTrue(upload.closed)

    def test_upload_route_rejects_file_over_size_limit(self):
        with patch.object(app, "MAX_UPLOAD_SIZE", 8):
            upload = FakeUploadFile(filename="calendar.pdf", contents=b"%PDF" + b"x" * 9)
            with self.assertRaises(HTTPException) as error:
                run_async_without_loop(app.upload_document(upload))
        self.assertEqual(error.exception.status_code, 413)
        self.assertTrue(upload.closed)

    def test_upload_route_sanitizes_windows_file_path(self):
        with patch.object(app, "index_pdf", return_value={"filename": "calendar.pdf"}) as index:
            upload = FakeUploadFile(filename=r"C:\private\calendar.pdf", contents=b"%PDF-1.7")
            result = run_async_without_loop(app.upload_document(upload))
        self.assertEqual(result["filename"], "calendar.pdf")
        self.assertEqual(index.call_args.args[0], "calendar.pdf")
        self.assertTrue(upload.closed)

    def test_question_length_is_limited(self):
        with self.assertRaises(ValidationError):
            app.Question(question="x" * 1_001)

    def test_upload_spinner_has_a_real_hidden_css_state(self):
        css = (Path(__file__).parents[1] / "web" / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".upload-progress[hidden]{display:none}", css)

    def test_ui_does_not_show_raw_source_cards(self):
        script = (Path(__file__).parents[1] / "web" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("source-card", script)

    def test_public_render_config_does_not_add_login_credentials(self):
        config = (Path(__file__).parents[1] / "render.yaml").read_text(encoding="utf-8")
        self.assertNotIn("APP_USERNAME", config)
        self.assertNotIn("APP_PASSWORD", config)

    def test_multiple_documents_can_be_indexed(self):
        for filename, page_text in (
            ("autumn.pdf", "Autumn semester classes begin August 24, 2026."),
            ("spring.pdf", "Spring semester classes begin January 19, 2027."),
        ):
            with patch.object(rag, "PdfReader", return_value=FakeReader([FakePage(page_text)])):
                rag.index_pdf(filename, b"%PDF-fake")
        self.assertEqual({doc["filename"] for doc in rag.get_documents()}, {"autumn.pdf", "spring.pdf"})

    def test_deadline_question_uses_calendar_dates_without_calling_gemini(self):
        self.add_passage("Project report deadline is October 12, 2026.")
        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(rag.requests, "post", side_effect=AssertionError("This date lookup should stay local")):
            result = rag.answer_question("What is the deadline to finish my project?")
        self.assertIn("October 12, 2026", result["answer"])
        self.assertIn("submit early", result["answer"])

    def test_gemini_success_is_trimmed_to_two_sentences_and_no_source_labels(self):
        self.add_passage("Autumn classes begin Monday, August 24, 2026.")

        class GoodResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "Classes begin August 24, 2026. [Source 1] Bring your ID. Third sentence."}]}}]}

        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(rag.requests, "post", return_value=GoodResponse()):
            result = rag.answer_question("When does Autumn semester begin?")
        self.assertNotIn("Source", result["answer"])
        self.assertNotIn("Third sentence", result["answer"])

    def test_unexpected_gemini_response_uses_grounded_fallback(self):
        self.add_passage("Autumn classes begin Monday, August 24, 2026.")

        class EmptyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": []}

        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(rag.requests, "post", return_value=EmptyResponse()):
            result = rag.answer_question("When does Autumn semester begin?")
        self.assertIn("August 24, 2026", result["answer"])

    def test_passage_deletion_only_removes_the_requested_calendar(self):
        self.add_passage("Project submission deadline is October 12, 2026.")
        connection = rag.connect_database()
        connection.execute(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
            ("id-second", "doc-2", "other.pdf", 1, "Spring classes begin January 19, 2027.", "[]"),
        )
        connection.commit()
        connection.close()
        self.assertTrue(rag.remove_document("doc-1"))
        self.assertEqual([doc["filename"] for doc in rag.get_documents()], ["other.pdf"])

    def test_invalid_pdf_bytes_are_rejected_cleanly(self):
        with self.assertRaisesRegex(ValueError, "could not be opened"):
            rag.index_pdf("invalid.pdf", b"not a PDF")


if __name__ == "__main__":
    unittest.main()
