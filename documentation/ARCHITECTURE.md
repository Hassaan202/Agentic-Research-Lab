# Architecture Explanation

## 🔍 Why Three Separate Files?

Each file has a **single, specific responsibility**. This is called **"Separation of Concerns"** - a key software engineering principle.

---

## 📊 The Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSOR                        │
│                  (document_processor.py)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  This is the ORCHESTRATOR - it coordinates           │   │
│  │  the other two modules                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ uses
                          ▼
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐          ┌──────────────────┐
│ DOCUMENT LOADER  │          │  VECTOR STORE    │
│                  │          │                  │
│ document_loader  │          │ vector_store.py  │
│      .py         │          │                  │
│                  │          │                  │
│ Job:             │          │ Job:             │
│ • Read files     │          │ • Store          │
│ • Parse PDF/TXT  │          │   embeddings     │
│ • Split text     │          │ • Search         │
│   into chunks    │          │   similar docs   │
│                  │          │                  │
└──────────────────┘          └──────────────────┘
        │                                 │
        │ chunks                          │ embeddings
        │                                 │
        └───────────┬─────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  ChromaDB     │
            │  Database     │
            └───────────────┘
```

---

## 📁 File Responsibilities

### 1️⃣ `document_loader.py` - **File Reader & Text Processor**
**What it does:**
- ✅ Reads PDF, TXT, DOCX files from `uploaded_documents/` folder
- ✅ Extracts text from files
- ✅ Splits text into smaller chunks (for embedding)
- ❌ **Does NOT** handle embeddings
- ❌ **Does NOT** handle database storage

**Think of it as:** A librarian who reads books and prepares pages

---

### 2️⃣ `vector_store.py` - **Database Manager**
**What it does:**
- ✅ Takes text chunks (from DocumentLoader)
- ✅ Converts them to embeddings (using Google Gemini embeddings)
- ✅ Stores embeddings in ChromaDB
- ✅ Searches for similar documents
- ❌ **Does NOT** read files
- ❌ **Does NOT** parse PDFs

**Think of it as:** A database that stores and searches documents

---

### 3️⃣ `document_processor.py` - **Orchestrator (The Boss)**
**What it does:**
- ✅ **Uses** DocumentLoader to get chunks
- ✅ **Uses** VectorStore to store chunks
- ✅ Coordinates the entire pipeline
- ✅ Provides a simple interface: `process_documents()`

**Think of it as:** The manager who coordinates the librarian and database

---

## 🔄 How They Work Together

### Step-by-Step Flow:

```python
# 1. You call the processor
processor = DocumentProcessor()

# 2. Processor uses DocumentLoader
chunks = processor.document_loader.process_all()
#    ↓
#    DocumentLoader reads files and creates chunks
#    Returns: List of text chunks

# 3. Processor uses VectorStore
processor.vector_store.add_documents(chunks)
#    ↓
#    VectorStore converts chunks to embeddings
#    Stores them in ChromaDB

# 4. Done! Documents are now searchable
```

---

## 💡 Why This Design?

### ✅ **Benefits:**

1. **Modularity**: Each file does one thing well
2. **Reusability**: You can use DocumentLoader without VectorStore
3. **Testability**: Test each component separately
4. **Maintainability**: Fix bugs in one place
5. **Flexibility**: Swap components (e.g., use different vector DB)

### ❌ **Without Separation:**

If everything was in one file:
- Hard to test
- Hard to reuse
- Hard to maintain
- Can't swap components

---

## 🎯 Real-World Analogy

**Restaurant Kitchen:**

- **DocumentLoader** = Prep Cook (cuts vegetables, prepares ingredients)
- **VectorStore** = Head Chef (cooks, stores food)
- **DocumentProcessor** = Restaurant Manager (coordinates everything)

Each has a specific job, but they work together!

---

## 🔧 When to Use Each File Directly

### Use `DocumentLoader` directly when:
- You just want to read and chunk documents
- You don't need to store embeddings
- You're testing document parsing

### Use `VectorStore` directly when:
- You already have text chunks
- You just want to search existing documents
- You're testing search functionality

### Use `DocumentProcessor` when:
- You want the complete pipeline (most common case)
- You're building the application
- You want the simplest interface

### Use `RAGPipeline` when:
- You want to ask questions about your documents
- You've already processed documents
- You want answers with source citations

---

## 📝 Example: Direct Usage

```python
# Using DocumentLoader alone
from src.document_loader import DocumentLoader
loader = DocumentLoader()
chunks = loader.process_all()
# Now you have chunks, but they're not stored yet

# Using VectorStore alone
from src.vector_store import VectorStore
store = VectorStore()
# Assume chunks already exist
store.add_documents(chunks)
results = store.similarity_search("machine learning")

# Using DocumentProcessor (easiest)
from src.document_processor import DocumentProcessor
processor = DocumentProcessor()
processor.process_documents()  # Does everything!

# Using RAGPipeline (for questions)
from src.rag_pipeline import RAGPipeline
rag = RAGPipeline()
result = rag.answer_question("What is machine learning?")
print(result['answer'])
```

---

---

## 4️⃣ `rag_pipeline.py` - **Question-Answering System**

**What it does:**
- ✅ Retrieves relevant document chunks using VectorStore
- ✅ Builds context from retrieved chunks
- ✅ Generates answers using Google Gemini LLM (via LangChain wrapper)
- ✅ Returns answers with source citations
- ❌ **Does NOT** process new documents
- ❌ **Does NOT** modify the vector store

**Think of it as:** A research assistant that answers questions using your document collection

**Technical Details:**
- Uses `gemini-2.5-flash` model
- Uses LangChain's `ChatGoogleGenerativeAI` wrapper
- Retrieves top-k similar documents (default: 5)
- Generates answers based on retrieved context

---

## ✅ Summary

| File | Responsibility | Input | Output |
|------|---------------|-------|--------|
| `document_loader.py` | Read & chunk files | Files in folder | Text chunks |
| `vector_store.py` | Store & search | Text chunks | Embeddings in DB |
| `document_processor.py` | Orchestrate | Nothing | Everything! |
| `rag_pipeline.py` | Answer questions | Questions | Answers + sources |

**They work together, but each has a clear, separate job!**

---

## 🔄 Complete Pipeline Flow

```
1. Document Processing (document_processor.py)
   ├─→ DocumentLoader: Reads files → Creates chunks
   └─→ VectorStore: Stores chunks → Creates embeddings

2. Question Answering (rag_pipeline.py)
   ├─→ VectorStore: Searches for similar chunks
   ├─→ Context Building: Combines retrieved chunks
   └─→ LLM (LangChain): Generates answer from context
```

