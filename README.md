# Agentic Research Lab

Agentic AI that turns a folder of research papers (PDF/TXT/DOCX) into a clear, citation‑grounded report in minutes.

## What It Does
Ingest your papers → build a local vector index → run a 5‑stage agent chain (Researcher → Reviewer → Synthesizer → Questioner → Formatter) → output an executive summary, key findings, critiques, hypotheses, and prioritized research questions with source citations.

## Why It Helps
Manual literature review is slow and easy to miss patterns. This project accelerates early research exploration by:
- Extracting methods, results, limitations
- Highlighting strengths, weaknesses, biases
- Connecting findings across papers
- Surfacing gaps & generating testable hypotheses
- Producing a structured, traceable report (no free‑form hallucinations)

## Core Features
- Multi‑agent sequential workflow (5 specialized roles)
- Retrieval‑Augmented Generation (RAG) on your local documents (ChromaDB)
- Google Gemini models for grounded generation + embeddings
- Structured outputs with per‑item source citations
- CLI, LangGraph orchestration, and REST API
- Optional Next.js + Tailwind frontend

## Quick Start (Backend)
```bash
# 1. Create venv (macOS / zsh)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install Python deps (filename currently has a space)
pip install -r "requirements .txt"  # or rename to requirements.txt first

# 3. Add your Gemini API key in .env
printf "GOOGLE_API_KEY=your_key_here" > .env

# 4. Drop PDFs into uploaded_documents/
#    e.g. uploaded_documents/your-paper.pdf

# 5. Ingest & index
python src/document_processor.py

# 6. Run multi-agent workflow
python src/multi_agent_system.py --query "Comprehensive analysis of the documents"
```
Report saved to `research_report.txt`.

## Alternative Runs
- LangGraph orchestrator: `python src/langgraph_multiagent.py`
- REST API (FastAPI): `uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload`
- RAG Q&A: `python src/rag_pipeline.py --question "What benchmarks are used?" --k 6`

## Agent Roles (Simple)
| Agent | Purpose |
|-------|---------|
| Researcher | Extract key findings & contributions |
| Reviewer | Critique methods & note limitations |
| Synthesizer | Link findings & propose hypotheses |
| Questioner | Expose gaps & pose research questions |
| Formatter | Assemble the final structured report |

## Folder Highlights
- `uploaded_documents/` – put source PDFs here
- `src/document_processor.py` – ingest + index
- `src/multi_agent_system.py` – procedural chain
- `src/langgraph_multiagent.py` – graph orchestration
- `vector_db/` & `summaries_vector_db/` – persisted Chroma indices
- `summaries/` – generated paper summaries (txt/json)

## Docs
- Setup: `docs/SETUP.md`
- Technical Overview: `docs/TECHNICAL_OVERVIEW.md`

## Troubleshooting (Common)
- Missing API key: ensure `.env` contains `GOOGLE_API_KEY`.
- Empty results: verify PDFs exist, then re‑run ingestion.
- Rebuild from scratch:
```bash
rm -rf vector_db summaries_vector_db summaries
python src/document_processor.py --clear
```

## Roadmap Ideas
- Add evaluation / grounding scores
- Plug in alternative LLMs & embedding models
- Export to Markdown / HTML / JSON API

## License
MIT License © 2025 Muhammad Hassaan Raza. See [LICENSE](./LICENSE).

---
Built for rapid, explainable literature synthesis. Contributions welcome.
