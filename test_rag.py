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

    def test_chunker_preserves_calendar_table_line_breaks(self):
        chunks = rag.split_into_chunks("August 24, 2026\nFirst day of classes\nAutumn begins")
        self.assertEqual(chunks, ["August 24, 2026\nFirst day of classes\nAutumn begins"])

    def test_calendar_table_parser_rejoins_wrapped_cross_year_date_ranges(self):
        rows = rag._calendar_table_rows(
            "December 24,\n2026-January 1,\n2027\nWinter university closure\nNo classes.\n",
            rag.CALENDAR_DATE_PATTERN,
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("December 24", rows[0][0])
        self.assertIn("January 1", rows[0][0])
        self.assertEqual(rows[0][1], "Winter university closure")

    def test_winter_closure_is_recognized_as_a_calendar_question(self):
        self.assertTrue(rag._is_calendar_question("When is the winter university closure?"))

    def test_all_dates_query_detects_every_and_each_phrasings(self):
        for question in (
            "give each date",
            "give every dates present in it",
            "list all dates",
        ):
            with self.subTest(question=question):
                self.assertTrue(rag._asks_for_all_dates(question))
        self.assertFalse(rag._asks_for_all_dates("When is the deadline?"))
        self.assertFalse(rag._asks_for_all_dates("List every date when classes are not held"))

    def test_no_class_dates_question_uses_category_filtered_events(self):
        self.add_passage(
            "AUTUMN SEMESTER 2026\n"
            "September 7, 2026\nLabor Day\nNo classes; university offices are closed.\n"
            "September 8, 2026\nClasses resume\nRegular instruction continues.\n"
            "November 25-27, 2026\nThanksgiving recess\nNo classes.\n"
        )
        pages = [
            {"filename": filename, "page": page, "text": text}
            for _, filename, page, text in rag._all_calendar_pages()
        ]
        answer = rag._local_calendar_answer("List every date when classes are not held", pages)
        self.assertIn("September 7, 2026", answer)
        self.assertIn("November 25-27, 2026", answer)
        self.assertNotIn("September 8, 2026", answer)

    def test_all_dates_lists_full_calendar_with_event_titles_and_ranges(self):
        self.add_passage(
            "AUTUMN SEMESTER 2026\n"
            "August 24, 2026 First day of classes; Autumn semester instruction begins.\n"
            "DATE\nACADEMIC EVENT\nDETAILS\n"
            "August 19-21, 2026\nNew student orientation\nOrientation and advising for new students.\n"
            "September 4, 2026\nAdd/drop deadline\nLast day to add or drop a course.\n"
            "October 12-16,\n2026\nMidterm assessment week\nInstructors may schedule exams.\n"
            "November 6, 2026\nCourse withdrawal\ndeadline\nLast day to withdraw from an Autumn course.\n"
        )
        answer = rag.list_all_calendar_dates()
        self.assertIn("August 19-21, 2026 — New student orientation", answer)
        self.assertIn("August 24, 2026", answer)
        self.assertIn("September 4, 2026 — Add/drop deadline", answer)
        self.assertIn("October 12-16, 2026 — Midterm assessment week", answer)
        self.assertIn("November 6, 2026 — Course withdrawal deadline", answer)
        self.assertNotIn("DATE ACADEMIC EVENT DETAILS", answer)
        self.assertEqual(answer.count("August 24, 2026"), 1)

    def test_all_dates_question_bypasses_retrieval_and_gemini(self):
        self.add_passage(
            "AUTUMN SEMESTER 2026\n"
            "August 24, 2026\nFirst day of classes\n"
            "September 4, 2026\nAdd/drop deadline\n",
        )
        with patch.object(rag, "retrieve", side_effect=AssertionError("Must inspect the full corpus")), patch.object(
            rag.requests, "post", side_effect=AssertionError("Must not call Gemini")
        ):
            result = rag.answer_question("give every dates present in it")
        self.assertIn("August 24, 2026", result["answer"])
        self.assertIn("September 4, 2026", result["answer"])
        self.assertEqual(result["sources"][0]["filename"], "calendar.pdf")

    def test_follow_up_suggestions_match_the_question_topic(self):
        suggestions = rag.suggest_follow_up_questions("When are the Autumn semester exams?")
        self.assertEqual(len(suggestions), 3)
        self.assertTrue(all("Autumn" in question for question in (suggestions[0], suggestions[-1])))
        self.assertTrue(any("deadlines" in question for question in suggestions))

    def test_chat_route_returns_clickable_follow_up_question_data(self):
        with patch.object(app, "answer_question", return_value={"answer": "Classes begin August 24.", "sources": []}):
            response = app.chat(app.Question(question="When does Autumn semester begin?"))
        self.assertEqual(len(response["suggestions"]), 3)
        self.assertTrue(all(isinstance(question, str) for question in response["suggestions"]))

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
        self.assertIn("Sorry, I couldn't find this information", answer)
        self.assertNotIn("September 4", answer)

    def test_calendar_follow_up_inherits_the_last_mentioned_semester(self):
        expanded = rag._contextualize_follow_up(
            "When does it end?",
            [{"role": "user", "content": "When does Autumn semester begin?"}],
        )
        self.assertIn("autumn semester", expanded.lower())

    def test_named_holiday_question_does_not_inherit_prior_semester(self):
        question = "when is diwali holidays"
        expanded = rag._contextualize_follow_up(
            question,
            [
                {"role": "user", "content": "When are Spring midterm assessments?"},
                {"role": "assistant", "content": "Spring midterms are March 8-12, 2027."},
            ],
        )
        self.assertEqual(expanded, question)

    def test_unsupported_named_holiday_returns_not_found_not_neighboring_dates(self):
        self.add_passage(
            "SPRING SEMESTER 2027\n"
            "February 5, 2027\nSpring tuition adjustment deadline\nLast day for eligible changes.\n"
            "March 15-19, 2027\nSpring recess\nNo classes.\n"
        )
        os.environ["GEMINI_API_KEY"] = "test-key"
        class NotFoundResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "Sorry, I couldn't find this information in the available academic calendar. Please check the official university website or contact the university administration."}]}}]}

        with patch.object(rag.requests, "post", return_value=NotFoundResponse()):
            result = rag.answer_question(
                "when is diwali holidays",
                history=[{"role": "assistant", "content": "Spring midterms are March 8-12, 2027."}],
            )
        self.assertIn("Sorry, I couldn't find this information", result["answer"])
        self.assertNotIn("February 5", result["answer"])
        self.assertNotIn("March 15", result["answer"])

    def test_days_until_is_calculated_from_the_verified_calendar_date(self):
        sources = [{
            "filename": "calendar.pdf",
            "page": 1,
            "text": "Autumn classes begin Monday, August 24, 2026.",
        }]
        with patch.object(rag, "_today", return_value=__import__("datetime").date(2026, 7, 19)):
            answer = rag._days_until_answer("How many days until Autumn semester begins?", sources)
        self.assertIn("36 days", answer)
        self.assertIn("August 24, 2026", answer)

    def test_countdown_question_uses_full_calendar_and_selects_semester_end(self):
        self.add_passage(
            "AUTUMN SEMESTER 2026\n"
            "August 24, 2026\nFirst day of classes\n"
            "November 6, 2026\nCourse withdrawal deadline\n"
            "December 11, 2026\nLast day of classes\nFinal day of Autumn instruction.\n"
            "December 16-18, 2026\nFinal examinations\n",
        )
        os.environ.pop("GEMINI_API_KEY", None)
        with patch.object(rag, "_today", return_value=__import__("datetime").date(2026, 10, 19)), patch.object(
            rag.requests, "post", side_effect=AssertionError("Verified countdowns must stay local")
        ):
            answer = rag.answer_question("How many days remain until Autumn semester ends?")
        self.assertIn("53 days", answer["answer"])
        self.assertIn("December 11, 2026", answer["answer"])
        self.assertNotIn("November 6, 2026", answer["answer"])

    def test_unrelated_question_gets_a_polite_scope_reply(self):
        result = rag.answer_question("Can you write a Python game for me?")
        self.assertIn("designed primarily", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_table_extraction_keeps_event_attached_to_date(self):
        source = {
            "filename": "calendar.pdf",
            "page": 1,
            "text": "DATE EVENT August 24, 2026 First day of classes Autumn semester instruction begins. September 4, 2026 Add/drop deadline.",
        }
        answer = rag._local_calendar_answer("When do Autumn classes begin?", [source])
        self.assertIn("August 24, 2026", answer)
        self.assertIn("Autumn classes begin", answer)

    def test_final_exam_answer_keeps_date_range_and_excludes_midterms_and_grade_dates(self):
        source = {
            "filename": "calendar.pdf",
            "page": 1,
            "text": "Spring 2027\nDATE\nEVENT / DEADLINE\nDETAILS\nMarch 8-12, 2027\nSpring midterm assessment period\nInstructors may schedule midterm exams.\nMay 12-14, 2027\nSpring final examination period\nFinal exams follow the published timetable; check the student portal for room and time.\nMay 21, 2027\nSpring grades available\nFinal grades are available in the student portal.",
        }
        answer = rag._local_calendar_answer("When are spring final exams?", [source])
        self.assertIn("May 12-14, 2027", answer)
        self.assertIn("student portal", answer)
        self.assertNotIn("March 8", answer)
        self.assertNotIn("May 21", answer)
        self.assertNotIn("\n", answer.strip("\n").split(":", 1)[0])

    def test_autumn_start_answer_does_not_mix_in_spring(self):
        passages = [
            {"filename": "calendar.pdf", "page": 1, "text": "AUTUMN SEMESTER 2026 Autumn classes begin Monday, August 24, 2026. DATE ACADEMIC EVENT DETAILS August 24, 2026 First day of classes Autumn semester instruction begins. December 11, 2026 Last day of classes Final day of Autumn instruction."},
            {"filename": "calendar.pdf", "page": 2, "text": "SPRING SEMESTER 2027 Spring classes begin Tuesday, January 19, 2027. January 19, 2027 First day of classes Spring semester instruction begins. May 7, 2027 Last day of classes Final day of Spring instruction."},
        ]
        answer = rag._local_calendar_answer("When does Autumn semester begin?", passages)
        self.assertIn("August 24, 2026", answer)
        self.assertNotIn("January 19, 2027", answer)

    def test_semester_end_question_returns_end_dates_not_start_dates(self):
        passages = [
            {"filename": "calendar.pdf", "page": 1, "text": "AUTUMN SEMESTER 2026 Autumn classes begin Monday, August 24, 2026. December 11, 2026 Last day of classes Final day of Autumn instruction."},
            {"filename": "calendar.pdf", "page": 2, "text": "SPRING SEMESTER 2027 Spring classes begin Tuesday, January 19, 2027. May 7, 2027 Last day of classes Final day of Spring instruction."},
        ]
        answer = rag._local_calendar_answer("When does the semester end?", passages)
        self.assertIn("December 11, 2026", answer)
        self.assertIn("May 7, 2027", answer)
        self.assertNotIn("August 24, 2026", answer)
        self.assertNotIn("January 19, 2027", answer)

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

    def test_ui_renders_and_handles_follow_up_question_chips(self):
        script = (Path(__file__).parents[1] / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("YOU COULD ALSO ASK", script)
        self.assertIn("follow-up-question", script)
        self.assertIn("result.suggestions", script)

    def test_ui_displays_source_page_and_bolds_dates(self):
        script = (Path(__file__).parents[1] / "web" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("Source: ${citation}", script)
        self.assertIn("<strong>$&</strong>", script)

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
    def test_deadline_question_uses_gemini_first_when_configured(self):
        self.add_passage("Project report deadline is October 12, 2026.")
        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(rag.requests, "post", side_effect=AssertionError("This date lookup should stay local")):
        class GeminiResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "The project report is due October 12, 2026. Submit it before the deadline."}]}}]}

        with patch.object(rag.requests, "post", return_value=GeminiResponse()) as generate:
            result = rag.answer_question("What is the deadline to finish my project?")
        generate.assert_called_once()
        self.assertIn("October 12, 2026", result["answer"])
        self.assertIn("submit early", result["answer"])
        self.assertIn("Submit it", result["answer"])

    def test_calendar_overview_sends_full_document_to_gemini_after_search_miss(self):
        self.add_passage("Autumn classes begin August 24, 2026. Spring exams run May 12-14, 2027.")
        os.environ["GEMINI_API_KEY"] = "test-key"

        class GeminiResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "The calendar is organized by semester, with class dates, deadlines, breaks, and exam periods. It covers Autumn 2026 and Spring 2027."}]}}]}

        with patch.object(rag, "retrieve", return_value=[]), patch.object(
            rag.requests, "post", return_value=GeminiResponse()
        ) as generate:
            result = rag.answer_question("Summarize the overall structure of the uploaded academic calendar.")
        generate.assert_called_once()
        sent = generate.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("Autumn classes begin August 24, 2026", sent)
        self.assertIn("organized by semester", result["answer"])

    def test_calendar_answer_searches_all_pages_not_only_top_k_retrieval(self):
        for page in range(1, 8):
            text = (
                "SPRING SEMESTER 2027\n"
                + ("April 2, 2027\nCourse withdrawal deadline\nLast day to withdraw from a Spring course.\n"
                   if page == 7 else f"Page {page} general spring calendar information.\n")
            )
            self.add_passage(text, page=page)
        with patch.object(rag, "retrieve", return_value=[
            {"filename": "calendar.pdf", "page": 1, "text": "General spring information."}
        ]):
            result = rag.answer_question("When is the spring course withdrawal deadline?")
        self.assertIn("April 2, 2027", result["answer"])

    def test_days_before_comparative_question_uses_both_calendar_dates(self):
        self.add_passage(
            "SPRING SEMESTER 2027\n"
            "January 11, 2027\nSpring tuition and fees due\nPay by the posted deadline.\n"
            "January 19, 2027\nFirst day of classes\nSpring semester instruction begins.\n"
        )
        os.environ.pop("GEMINI_API_KEY", None)
        with patch.object(rag.requests, "post", side_effect=AssertionError("Date comparison stays local")):
            result = rag.answer_question("How many days before spring classes begin is tuition due?")
        self.assertIn("8 days before", result["answer"])

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
