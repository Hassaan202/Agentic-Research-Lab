# Quick Reference: Understanding the Three Files

## 🎯 One Sentence Summary

- **document_loader.py** = Reads files → Creates text chunks
- **vector_store.py** = Takes chunks → Stores as embeddings
- **document_processor.py** = Uses both → Complete pipeline
- **rag_pipeline.py** = Questions → Answers (using LangChain wrapper)

---

## 📋 What Each File Does

### 📄 document_loader.py
```
INPUT:  Files in uploaded_documents/ folder
        (paper1.pdf, paper2.txt, etc.)

PROCESS: 
  - Reads PDF/TXT files
  - Extracts text
  - Splits into chunks (1000 chars each)

OUTPUT: List of text chunks
        ["chunk 1 text...", "chunk 2 text...", ...]
```

### 🗄️ vector_store.py
```
INPUT:  Text chunks (from DocumentLoader)
        ["chunk 1 text...", "chunk 2 text...", ...]

PROCESS:
  - Converts chunks to embeddings (Google Gemini)
  - Stores embeddings in ChromaDB
  - Enables similarity search

OUTPUT: Documents stored in database
        Can search: store.similarity_search("query")
```

### 🎛️ document_processor.py
```
INPUT:  Nothing (or folder path)

PROCESS:
  1. Uses DocumentLoader → gets chunks
  2. Uses VectorStore → stores chunks

OUTPUT: Complete pipeline done!
        Documents are now searchable
```

### 🤖 rag_pipeline.py
```
INPUT:  Question from user
        "What is machine learning?"

PROCESS:
  1. Uses VectorStore → searches for similar chunks
  2. Builds context from retrieved chunks
  3. Uses LLM (LangChain wrapper, gemini-2.5-flash) → generates answer

OUTPUT: Answer with source citations
        {
          "answer": "Machine learning is...",
          "sources": [...]
        }
```

---

## 🔄 The Flow

```
Your Code
    │
    │ processor.process_documents()
    ▼
document_processor.py
    │
    ├─→ document_loader.py
    │      │
    │      │ Reads: paper1.pdf
    │      │ Returns: ["chunk1", "chunk2", ...]
    │      │
    │      └─→ Back to processor
    │
    └─→ vector_store.py
           │
           │ Takes: ["chunk1", "chunk2", ...]
           │ Stores: In ChromaDB
           │ Returns: Success
           │
           └─→ Back to processor
```

---

## 💻 Code Example

```python
# The complete flow in code:

# 1. DocumentLoader (reads files)
loader = DocumentLoader()
chunks = loader.process_all()
# chunks = ["text chunk 1", "text chunk 2", ...]

# 2. VectorStore (stores chunks)
store = VectorStore()
store.add_documents(chunks)
# Now chunks are stored as embeddings

# 3. DocumentProcessor (does both!)
processor = DocumentProcessor()
processor.process_documents()
# This internally does steps 1 and 2
```

---

## ✅ When to Use Which

| Use Case | Use This File |
|----------|--------------|
| Just want to read files | `document_loader.py` |
| Just want to store/search | `vector_store.py` |
| Complete pipeline (recommended) | `document_processor.py` |
| Want to ask questions | `rag_pipeline.py` |

---

## 🎓 Key Concept

**Separation of Concerns:**
- Each file does ONE thing
- They work together
- Easy to test and maintain

**Think of it like:**
- DocumentLoader = Librarian (reads books)
- VectorStore = Library Database (stores books)
- DocumentProcessor = Library Manager (coordinates both)
- RAGPipeline = Research Assistant (answers questions using books)

---

## 📚 More Details

See:
- `ARCHITECTURE.md` - Detailed architecture explanation
- `UNDERSTANDING_THE_CODE.md` - Step-by-step explanation
- `RAG_PIPELINE.md` - RAG pipeline documentation

