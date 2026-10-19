# Campus Calendar — Academic Calendar Assistant

A small, public website that searches uploaded academic-calendar PDFs and answers questions about dates. It uses retrieval-augmented generation (RAG): the app retrieves calendar passages first, then uses those passages to answer. On Render, retrieval is local BM25 search and Gemini is used only to write the answer. If Gemini is unavailable, a short answer is built from matching calendar text.

## Important limits before deployment

- Render Free sleeps after 15 minutes without traffic. The first visit after sleep can take about a minute.
- Render Free has an ephemeral filesystem. Uploaded calendars and the local SQLite database are lost when the service sleeps, restarts, or redeploys. Users may need to upload calendars again.
- This is one shared public workspace: visitors can see and remove calendars uploaded to the site. Do not upload private or sensitive documents.
- Gemini's free API has changing quotas and outages. The app falls back to retrieved calendar text when answer generation fails, but no free hosting or API can guarantee uninterrupted service.
- If a calendar does not contain a requested project deadline, the assistant says it is not listed rather than inventing a date.

For persistent files, a paid Render service with a persistent disk is needed. A persistent database is another option, but free database plans can have their own limits and expiration policies.

## Files

```text
academic-calendar-assistant/
├── .env.example       # Example local settings; never put a real key here
├── .gitignore         # Keeps secrets and local data out of Git
├── README.md          # This guide
├── app.py             # FastAPI website and upload/chat routes
├── rag.py             # PDF extraction, retrieval, and grounded answers
├── render.yaml        # Public Render Free deployment settings
├── requirements.txt   # Python packages
├── tests/
│   └── test_rag.py    # Offline reliability checks
└── web/
    ├── app.js         # Upload, chat, timeout, and progress behavior
    ├── index.html     # Website structure
    └── styles.css     # Website styling
```

## Run locally with Ollama

1. Open this folder in VS Code and open **Terminal → New Terminal**.
2. Keep Ollama running and download the models once:

   ```powershell
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

3. Start the website:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app:app --reload
   ```

4. Open `http://127.0.0.1:8000` and upload a text-based academic calendar PDF. Image-only scans need OCR first.

## Deploy on Render Free

1. Upload the project files to the top level of a GitHub repository. The `render.yaml` file must be beside `app.py`.
2. In Render, create a **Blueprint** from that repository.
3. When asked for `GEMINI_API_KEY`, enter the key from Google AI Studio. Keep it in Render's secret environment settings; never commit it to GitHub.
4. Wait for the service to show **Live**, then open its public `onrender.com` link.

The blueprint intentionally has no website username or password. Anyone with the link can use the shared site and shared calendar library.

## Test the project

Run the automated checks from the project folder:

```powershell
python -m unittest discover -s tests -v
```

The 26 offline checks cover short and long text chunking, PDF extraction errors, scanned/locked/oversized PDFs, retrieval relevance, date formats and table text, missing deadlines, Gemini outages and malformed replies, upload validation, document deletion, and the upload progress indicator.

## Settings

- Local Ollama is the default provider. For Render, `render.yaml` sets `AI_PROVIDER=gemini`.
- `GEMINI_API_KEY` belongs in Render's environment settings or a private local `.env` file. Never expose it in frontend code or a public repository.
- PDFs must be unlocked, text-readable, no larger than 20 MB, and no more than 500 pages.
- The local database is created at `data/calendar.db`.
