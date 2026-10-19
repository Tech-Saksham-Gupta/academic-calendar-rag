# Campus Calendar — Academic Calendar Assistant

A responsive website that answers questions from academic calendar PDFs and shows the pages used as evidence. It uses **RAG (retrieval-augmented generation)**: PDF pages are split into passages, converted into meaning-based embeddings, searched for relevant passages, and passed to an AI model to write an answer with sources.

## Two ways to run it

| Where | AI provider | Cost | Important detail |
|---|---|---|---|
| Your computer | Ollama | No API fee | Uses your computer's memory, storage, and electricity. |
| Render Free | Gemini API free tier | Can be $0 within the free limits | The service sleeps when idle, and uploaded calendars can disappear when it restarts. Google's free tier may use submitted content to improve its products. |

Gemini's free quota and available models can change. Keep the Google project on its free tier and do not enable billing if you want to avoid API charges. Do not upload private or sensitive records to the Gemini free tier.

## Project files

```text
academic-calendar-assistant/
├── .env.example       # Example settings; never put a real key here
├── .gitignore         # Keeps secrets and local database out of Git
├── README.md          # This guide
├── app.py             # Website server and API routes
├── rag.py              # PDF reading, embeddings, retrieval, and answers
├── render.yaml         # Free Render deployment setup
├── requirements.txt    # Python packages
└── web/
    ├── app.js          # Upload, chat, and source interactions
    ├── index.html      # Website page structure
    └── styles.css      # Website design
```

## Run locally with Ollama

You already have Python and the Python packages installed. Open the `academic-calendar-assistant` folder in VS Code, then open **Terminal → New Terminal**. Keep Ollama running, and download both models once:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

If PowerShell says `ollama` is not recognized, use the full path (as you did earlier):

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2:3b
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull nomic-embed-text
```

Then start the website:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser. Upload a text-based academic calendar PDF and ask a question. Scanned image-only PDFs need OCR before this app can read them.

## Deploy on Render Free

This option runs the website on Render Free and uses Google's Gemini API free tier for embeddings and answers. Ollama is not needed on Render.

### 1. Create a free Gemini API key

1. Open [Google AI Studio API keys](https://aistudio.google.com/apikey) and sign in.
2. Create an API key for a free-tier project. Do not link billing if you want to keep API use free.
3. Keep the key private. Never paste it into `rag.py`, `app.py`, or a public GitHub repository.

Google's [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing) lists which models have free-tier use. The currently configured models are `gemini-2.5-flash-lite` for answers and `gemini-embedding-2` for retrieval embeddings. Free-tier limits and availability may change.

### 2. Put the project on GitHub

1. Create a new GitHub repository.
2. Upload the **contents** of the `academic-calendar-assistant` folder to the repository's top level. `render.yaml` should be visible at the top level, beside `app.py`.
3. Do not upload your `.venv` folder or any `.env` file with a real key.

### 3. Create the Render service

1. Sign in to [Render](https://render.com).
2. Choose **New → Blueprint** and connect your GitHub repository.
3. Review the service. The provided `render.yaml` selects the **Free** plan and Singapore region.
4. When Render asks for `GEMINI_API_KEY`, paste your key there. Render stores it as a secret environment value.
5. Create the service and wait for the first deployment to finish.
6. Open the `onrender.com` link. The username is `calendar-admin`; find the generated password on the service's **Environment** page.

### Free hosting limits

- Render Free sleeps after 15 minutes without incoming requests. The first visit after that can take about a minute to load.
- Render Free storage is temporary. Uploaded calendar passages are erased if the service restarts, sleeps, or redeploys. Upload your calendar again when it is missing.
- Google Gemini free API use has rate and usage limits. If you reach one, wait for the quota to reset or use local Ollama instead.
- Google's pricing page says content sent through the free tier may be used to improve Google products. Use local Ollama for documents you do not want to send to a hosted AI provider.

## How the RAG flow works

1. **Extract:** `pypdf` reads selectable text from each page of the PDF.
2. **Split:** the app divides text into overlapping passages and keeps the original page number.
3. **Embed:** Ollama or Gemini converts each passage into a numeric meaning vector.
4. **Retrieve:** the app embeds the question and compares vectors to find the most relevant passages.
5. **Generate:** the chosen model gets only the question and retrieved evidence. The website shows the answer and source page cards.

## Local files and settings

- The local SQLite database is created at `data/calendar.db` after the first PDF upload.
- Copy `.env.example` to `.env` only if you want to change settings. For local Ollama, the default `AI_PROVIDER=ollama` works without an API key.
- For Gemini locally, set `AI_PROVIDER=gemini` and put your key in `GEMINI_API_KEY` in your private `.env` file. Do not upload `.env` to GitHub.
- The upload limit is 20 MB per PDF. The app stores extracted passages and their embeddings, not the original PDF file.
