# Technical Implementation Guide

A deep dive into how the system is built: modules, data flow, dependencies, and extension points. Use this to modify or extend agents, swap models, or integrate new stores.

---

## Architecture at a glance

- Retrieval‑Augmented Generation (RAG) at the core: a shared pipeline retrieves context via ChromaDB, and Gemini models generate grounded outputs.
- Two orchestrations are provided:
  1) Procedural orchestrator (`src/multi_agent_system.py`)
  2) LangGraph graph orchestrator with memory/checkpointing (`src/langgraph_multiagent.py`)
- Shared components for ingestion and retrieval:
  - `DocumentLoader` → parse files, chunk text
  - `VectorStore` → embed and index chunks in ChromaDB
  - `RAGPipeline` → retrieve top‑k chunks and build prompts

---

## Key modules

- `src/document_loader.py`
  - Loads PDF/TXT/DOCX using LangChain community loaders
  - Splits text with `RecursiveCharacterTextSplitter`
  - Provides two flows:
    - `process_all()` → returns chunked `Document`s
    - `process_all_with_full_text()` → plus full‑text per file for summarization

- `src/vector_store.py`
  - Embeddings: `GoogleGenerativeAIEmbeddings` (model `models/text-embedding-004` by default)
  - Storage/Retrieval: `langchain-chroma` wrapper over ChromaDB (persisted under `vector_db/`)
  - Methods: `add_documents`, `similarity_search`, `similarity_search_with_score`, `get_collection_info`, `delete_collection`

- `src/rag_pipeline.py`
  - Composes the retrieval and LLM generation path
  - Forces LLM to `gemini-2.5-flash` via `ChatGoogleGenerativeAI` (reliable config)
  - `answer_question(question, k, return_sources)` returns:
    - `answer`: LLM output grounded in retrieved chunks
    - `sources`: file, page, chunk index, preview (if requested)
  - Also exposes `get_vector_store_stats()` and `get_model_info()`

- `src/summarizer_agent.py`
  - Lightweight single‑agent that extracts structured JSON metadata from a paper’s full text
  - Used by `DocumentProcessor` to create summary text and JSON, and to index summaries

- `src/document_processor.py`
  - Orchestrates ingest → chunk → embed → index
  - Optionally generates summaries and stores them as: text files (under `summaries/`) and vector entries (under `summaries_vector_db/`)
  - Returns rich stats after processing

- `src/agents.py` (Procedural agents)
  - BaseAgent: sets up Gemini chat model and shared RAG pipeline
  - Researcher → Reviewer → Synthesizer → Questioner → Formatter
  - Each agent:
    - Pulls fresh context using `RAGPipeline.answer_question()`
    - Produces structured, citation‑anchored output
    - Passes artifacts forward as a plain Python dict

- `src/multi_agent_system.py` (Procedural orchestrator)
  - Instantiates a shared `RAGPipeline` and all agents with tuned temperatures
  - Runs sequentially and aggregates results into a final report
  - CLI options for query, model, temperature, and output file

- `src/langgraph_multiagent.py` (LangGraph orchestrator)
  - Defines `ResearchState` and stateful nodes for each agent
  - Uses `StateGraph` + `MemorySaver` for routing, checkpointing, and a collaboration log
  - Provides `run_research_analysis(config)` and `create_research_graph()` helpers

- `src/api_server.py` (FastAPI backend)
  - Async API to kick off a background research run and fetch status/results
  - Endpoints:
    - `POST /api/research/start` → returns `thread_id`
    - `GET /api/research/status/{thread_id}` → in‑progress status
    - `GET /api/research/result/{thread_id}` → full result when done
    - `GET /api/research/collaboration/{thread_id}` → agent log
    - `GET /api/research/list` → list all threads

---

## Data flow

1. Ingestion
   - `DocumentProcessor` loads files from `uploaded_documents/`, splits into chunks
   - `VectorStore` indexes chunks into ChromaDB under `vector_db/`
   - Optionally, `SummarizerAgent` creates summary text and JSON, which are indexed into `summaries_vector_db/`

2. Retrieval and generation
   - Agents call `RAGPipeline.answer_question()` with a focused query per step
   - RAG retrieves top‑k chunks and composes a context‑rich prompt
   - Gemini (via LangChain) generates grounded text

3. Orchestration
   - Procedural: Each agent consumes the previous agent’s output and emits structured fields
   - LangGraph: State machine routes between nodes; includes collaboration log and final report

Outputs
- Terminal report and `research_report.txt`
- Summaries under `summaries/` (txt/json)
- Vector DBs under `vector_db/` and `summaries_vector_db/`

---

## Configuration

Environment
- `.env` at repo root: `GOOGLE_API_KEY=<key>`

Models
- Chat LLM: `gemini-2.5-flash` (enforced in RAG; agents initialize ChatGoogleGenerativeAI directly)
- Embeddings: `models/text-embedding-004`

Retrieval
- Default k per agent ranges 5–10; override via `RAGPipeline(..., max_retrieval_docs=K)` or agent `retrieve_context(..., k=K)`

Collections
- Documents: `research_documents` in `vector_db/`
- Summaries: `document_summaries` in `summaries_vector_db/`

---

## Running modes

- Procedural CLI: `python src/multi_agent_system.py --query "..."`
- LangGraph script: `python src/langgraph_multiagent.py`
- REST API: `uvicorn src.api_server:app --reload`
- RAG REPL: `python src/rag_pipeline.py` (interactive Q&A over your indexed corpus)

---

## Extending the system

Add a new agent (procedural)
- Create a class inheriting `BaseAgent` in `src/agents.py`
- Implement `process(self, input_data: Dict, query: str) -> Dict`
- Add it to `MultiAgentResearchSystem` sequence in `src/multi_agent_system.py`

Add a new node (LangGraph)
- Define a node function in `src/langgraph_multiagent.py`
- Update `ResearchState` TypedDict with new fields
- Wire edges in `create_research_graph()` and recompute routes

Swap embedding model or vector store
- Change `VectorStore` initialization to another embedding function
- Replace `langchain-chroma` with another retriever (e.g., FAISS, PGVector)

Tighten parsing and structure
- Improve section extractors in `agents.py` (`_extract_*` methods)
- Constrain outputs with JSON‑format prompts and pydantic validation

Guardrails and evaluation
- Add assertions on minimum `num_sources` per step
- Log retrieved chunks and citation IDs to verify grounding
- Integrate unit tests for prompt formatting and parsing logic

---

## API contract (selected endpoints)

- POST /api/research/start
  - body: `{ "thread_id"?: string }`
  - response: `{ thread_id, status: "started", current_step: 0, message }`

- GET /api/research/status/{thread_id}
  - response: `{ thread_id, status, current_step, message }`

- GET /api/research/result/{thread_id}
  - response: full aggregated object with researcher/reviewer/synthesizer/questioner/formatter fields

Notes
- Results are stored in‑memory (dicts). Persist to Redis/DB for durability.

---

## Frontend notes

- Next.js 14 with Tailwind UI components
- Scripts: `npm run dev|build|start`
- CORS is already enabled in the API for localhost:3000/3001

---

## Known constraints and caveats

- Model selection: Even if you pass a different LLM name, `RAGPipeline` enforces `gemini-2.5-flash` to avoid incompatibilities
- Requirements filename includes a space (`requirements .txt`); install with quotes or rename
- Large PDFs: summarizer truncates to a conservative character budget; tune in `summarizer_agent.py`
- Local vector DBs: delete `vector_db/` and `summaries_vector_db/` to rebuild cleanly

---

## Repository map (selected)

- `src/`
  - `agents.py` — procedural agents
  - `multi_agent_system.py` — procedural orchestrator
  - `langgraph_multiagent.py` — LangGraph graph + memory
  - `rag_pipeline.py` — retrieval + prompt + LLM answer
  - `document_loader.py` — file loading and chunking
  - `document_processor.py` — ingestion orchestration
  - `vector_store.py` — embeddings + Chroma
  - `summarizer_agent.py` — summary generator
- `uploaded_documents/` — place your PDFs here
- `vector_db/`, `summaries_vector_db/` — persisted indices
- `summaries/` — human‑readable summary exports

---

## Quick recipes

- Reset everything and re‑ingest
```bash
rm -rf vector_db summaries_vector_db summaries
python src/document_processor.py --clear
```

- Ask ad‑hoc questions over the corpus (RAG REPL)
```bash
python src/rag_pipeline.py --question "What benchmarks are commonly used?" --k 6
```

- Run the multi‑agent workflow on a topic
```bash
python src/multi_agent_system.py --query "State of the art in few-shot fine-tuning"
```

That’s the core. Modify prompts, tune temperatures, or add agents—everything is modular and locally persisted for fast iteration.
