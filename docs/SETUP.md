# Setup Guide

Get from raw PDFs to a citation‑grounded, AI‑generated research report. This guide covers backend ingestion, running the multi‑agent workflow, the API, and the frontend.

---

## 1) Prerequisites

- Python 3.10+ (3.12 works)
- Node.js 18+ (for the frontend)
- Google API Key for Gemini
  - Create one at: https://makersuite.google.com/app/apikey

Recommended:
- Python virtualenv (venv)
---

## 2) Clone and create a Python environment

```bash
# Clone the repo
git clone <your-repo-url>
cd Agentic-Research-Lab

# Create & activate a virtual environment (zsh/macOS)
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3) Install Python dependencies

Note: the requirements filename contains a space. Either quote the filename or rename it first.

```bash
# Option A: quote the filename
pip install -r "requirements .txt"

# Option B: rename then install
mv "requirements .txt" requirements.txt
pip install -r requirements.txt
```

---

## 4) Configure environment variables

Create a `.env` file in the repository root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

All agents and pipelines load this via `python-dotenv`.

---

## 5) Add your documents

Place PDFs, TXT, or DOCX files into `uploaded_documents/`:

```
uploaded_documents/
  your-paper-1.pdf
  your-paper-2.pdf
```

Supported types: PDF, TXT, DOCX.

---

## 6) Ingest documents (chunk → embed → index)

This builds local ChromaDB indices for both documents and (optionally) summaries.

```bash
# Process and index documents
python src/document_processor.py

# Useful options:
#   --clear          Clear and rebuild vector DBs
#   --folder PATH    Use a custom documents folder
#   --chunk-size N   Chunk size (default 1000)
#   --chunk-overlap N  Overlap (default 200)
#   --no-summaries   Skip generating summaries
```

Outputs:
- Document vector DB: `vector_db/`
- Summary vector DB: `summaries_vector_db/`
- Human‑readable summaries (if enabled): `summaries/*.txt` and `summaries/*.json`

---

## 7) Run the Multi‑Agent CLI workflow

Procedural orchestrator that runs Researcher → Reviewer → Synthesizer → Questioner → Formatter.

```bash
python src/multi_agent_system.py --query "Provide a comprehensive analysis of the documents"
```

Notes:
- A final report prints to the console and saves to `research_report.txt` (unless `--no-save`).
- The script currently sets a default query internally; feel free to edit `src/multi_agent_system.py` to change it.

Other options:
- `--vector-db PATH` (default: `vector_db`)
- `--collection NAME` (default: `research_documents`)
- `--model NAME` (internally normalized to `gemini-2.5-flash` for reliability)
- `--temperature FLOAT` (default: 0.3)
- `--output FILE` (default: `research_report.txt`)

---

## 8) Run the LangGraph workflow (alternative)

Graph‑based orchestrator with checkpointing and a collaboration log.

```bash
python src/langgraph_multiagent.py
```

On success, it prints the collaboration log and writes the final report to `research_report.txt`.

---

## 9) Start the REST API (FastAPI)

Run with Uvicorn (recommended):

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

- Health check: http://localhost:8000/health
- Start research (background job):
  ```bash
  curl -X POST http://localhost:8000/api/research/start \
       -H 'Content-Type: application/json' \
       -d '{}'
  ```
- Check status: `GET /api/research/status/{thread_id}`
- Get result: `GET /api/research/result/{thread_id}`

Note: The `__main__` block in `src/api_server.py` references a different module name; prefer the `uvicorn` command above.

---

## 10) Frontend (Next.js + Tailwind)

The frontend is optional and can run independently.

```bash
cd frontend
npm install
npm run dev
```

- Opens at http://localhost:3000
- Ensure the API is running on http://localhost:8000 if the UI calls the backend
- Requires Node 18+

---

## 11) Troubleshooting

- GOOGLE_API_KEY not found
  - Ensure `.env` exists at repo root and contains `GOOGLE_API_KEY`
  - Reactivate your virtualenv or restart terminal

- No relevant documents found / empty outputs
  - Add PDFs to `uploaded_documents/`
  - Re‑ingest: `python src/document_processor.py`

- Model override warning
  - The system standardizes on `gemini-2.5-flash` for reliability; warnings are informational

- Reset indices and summaries
  ```bash
  rm -rf vector_db summaries_vector_db summaries
  python src/document_processor.py --clear
  ```
---

You’re ready—ingest your papers and generate explainable, citation‑grounded research reports in minutes.
