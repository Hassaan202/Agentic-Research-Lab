# Agentic Research Lab

Agentic AI that turns a folder of research papers (PDF/TXT/DOCX) into a clear, citation‑grounded report in minutes.

## What It Does
Ingest your papers → build a local vector index → run a 5‑stage agent chain (Researcher → Reviewer → Synthesizer → Questioner → Formatter) → output an executive summary, key findings, critiques, hypotheses, and prioritized research questions with source citations.

## Why It Helps
Manual literature review is slow and prone to missing patterns. This project accelerates early research exploration by:
- Extracting methods, results, and limitations
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
# 1. Create venv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install Python deps
pip install -r "requirements.txt" 

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

## Full Setup and Technical Details
- see `docs/SETUP.md` and `docs/TECHNICAL_OVERVIEW.md` for more details

## Agent Roles
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

## License
MIT License © 2025 Muhammad Hassaan Raza. See [LICENSE](./LICENSE).
