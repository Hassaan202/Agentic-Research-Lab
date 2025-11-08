# Research Agent - Agentic AI for Accelerated Research

A multi-agent AI system for collaborative research analysis and insight generation.

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory with your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Add Documents

Place your research papers, PDFs, or text files in the `uploaded_documents` folder.

**Supported formats:**
- PDF files (`.pdf`)
- Text files (`.txt`)
- Word documents (`.docx`, `.doc`)

### 4. Process Documents

Run the document ingestion pipeline:

```bash
# Basic usage
python src/document_processor.py

# Clear existing vector store and reprocess
python src/document_processor.py --clear

# Custom documents folder
python src/document_processor.py --folder path/to/documents

# Custom chunk settings
python src/document_processor.py --chunk-size 1500 --chunk-overlap 300
```

### 5. Ask Questions with RAG Pipeline

After processing documents, you can ask questions using the RAG pipeline:

```bash
# Interactive Q&A mode
python src/rag_pipeline.py

# Single question mode
python src/rag_pipeline.py --question "What is machine learning?"

# Custom model and settings (note: model will be overridden to gemini-2.5-flash)
python src/rag_pipeline.py --model gemini-2.5-flash --temperature 0.5 --k 10
```

### 6. Usage in Python

```python
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline

# Process documents
processor = DocumentProcessor()
result = processor.process_documents()

# Initialize RAG pipeline
rag = RAGPipeline(
    vector_db_path="vector_db",
    llm_model="gemini-2.5-flash",  # Uses LangChain wrapper
    temperature=0.7,
    max_retrieval_docs=5
)

# Ask questions
answer = rag.answer_question("What is the main topic?")
print(answer['answer'])
```

## 📁 Project Structure

```
.
├── uploaded_documents/          # Place your research papers/PDFs here
├── vector_db/                   # ChromaDB vector database (auto-generated)
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # Main document ingestion pipeline
│   ├── document_loader.py       # Document loading utilities
│   ├── vector_store.py          # Vector database management
│   └── rag_pipeline.py          # RAG pipeline for Q&A (uses LangChain wrapper)
├── documentation/               # Documentation files
│   ├── README.md                # This file
│   ├── ARCHITECTURE.md          # Architecture explanation
│   ├── UNDERSTANDING_THE_CODE.md # Code explanation
│   ├── QUICK_REFERENCE.md       # Quick reference guide
│   └── RAG_PIPELINE.md          # RAG pipeline documentation
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
└── README.md                    # Main README (if exists)
```

## 🔧 Features

- **Multi-format Support**: PDF, TXT, DOCX
- **Intelligent Chunking**: Recursive text splitting with configurable overlap
- **Vector Embeddings**: Google Gemini embeddings stored in ChromaDB
- **Similarity Search**: Fast semantic search across documents
- **RAG Pipeline**: Question-answering using retrieved context and LLM generation
- **Interactive Q&A**: Terminal interface for asking questions about documents
- **Metadata Tracking**: Source file tracking and chunk indexing
- **Persistent Storage**: Vector database persists between sessions

## 📊 How It Works

### Document Processing
1. **Document Loading**: Reads all supported files from `uploaded_documents` folder
2. **Text Splitting**: Splits documents into chunks (default: 1000 chars with 200 overlap)
3. **Embedding**: Generates embeddings using Google Gemini's embedding model
4. **Storage**: Stores embeddings in ChromaDB vector database

### RAG Pipeline
1. **Question Input**: User asks a question via terminal or Python API
2. **Retrieval**: System retrieves relevant document chunks using semantic search
3. **Context Building**: Retrieved chunks are combined into context
4. **Answer Generation**: Google Gemini LLM (via LangChain wrapper) generates answer based on context
5. **Source Citation**: Returns answer with source document references

**Note**: The RAG pipeline uses `gemini-2.5-flash` model with LangChain's `ChatGoogleGenerativeAI` wrapper. The model name will be automatically overridden to `gemini-2.5-flash` if a different model is specified.

## 🎯 Next Steps

This is the foundation for the multi-agent research system. Next steps:
- Implement multi-agent reasoning (Researcher, Critic, Synthesizer agents)
- Add agent collaboration and conversation flow
- Generate collective insight reports
- Create visualization dashboard

## 📝 Notes

- The vector database is stored locally in the `vector_db` folder
- Documents are chunked to optimize for both retrieval and context window limits
- First run will create the vector database; subsequent runs will append new documents
- Use `--clear` flag to start fresh (deletes existing vector database)

