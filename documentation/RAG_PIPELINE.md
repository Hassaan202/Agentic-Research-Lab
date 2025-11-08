# RAG Pipeline Documentation

## Overview

The RAG (Retrieval-Augmented Generation) pipeline is implemented in `src/rag_pipeline.py` and provides question-answering functionality over your document collection.

## Architecture

```
User Question
    ↓
RAGPipeline.answer_question()
    ↓
1. VectorStore.similarity_search()  → Retrieves relevant document chunks
    ↓
2. Context Building                  → Combines retrieved chunks
    ↓
3. LLM Generation (LangChain)        → Generates answer using gemini-2.5-flash
    ↓
4. Response with Sources            → Returns answer + source citations
```

## Key Components

### RAGPipeline Class

**Location**: `src/rag_pipeline.py`

**Responsibilities**:
- Retrieves relevant documents from vector store
- Builds context from retrieved chunks
- Generates answers using Google Gemini LLM (via LangChain wrapper)
- Returns answers with source citations

### LLM Integration

**Model**: `gemini-2.5-flash`

**API Method**: LangChain wrapper (`ChatGoogleGenerativeAI`)

**Why LangChain wrapper?**:
- Standardized interface
- Built-in error handling and retries
- Easy integration with LangChain ecosystem
- Consistent message formatting

## Usage

### Command Line

```bash
# Interactive mode
python src/rag_pipeline.py

# Single question
python src/rag_pipeline.py --question "What is the main topic?"

# Custom settings
python src/rag_pipeline.py --temperature 0.5 --k 10
```

### Python API

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    vector_db_path="vector_db",
    llm_model="gemini-2.5-flash",  # Will be used (other models overridden)
    temperature=0.7,
    max_retrieval_docs=5
)

# Ask a question
result = rag.answer_question("What is machine learning?")
print(result['answer'])
print(f"Sources: {result['sources']}")
```

## Model Configuration

**Current Model**: `gemini-2.5-flash`

**Important Notes**:
- The model name is automatically set to `gemini-2.5-flash` (working model)
- If you specify a different model, it will be overridden with a warning
- Uses LangChain's `ChatGoogleGenerativeAI` wrapper
- Temperature and other parameters are configurable

## Response Format

```python
{
    "answer": "The answer text...",
    "question": "Original question",
    "num_sources": 5,
    "sources": [
        {
            "source": "uploaded_documents/paper1.pdf",
            "page": 1,
            "chunk_index": 0,
            "content_preview": "Preview of chunk content..."
        },
        ...
    ]
}
```

## Interactive Commands

When running in interactive mode, you can use:

- `help` - Show available commands
- `stats` - Show vector store and model statistics
- `model` - Show current model information
- `quit` or `exit` - Exit the program
- Any other text - Ask a question

## How It Works

1. **Question Processing**: User asks a question
2. **Document Retrieval**: System searches vector store for relevant chunks (default: 5 chunks)
3. **Context Building**: Retrieved chunks are combined into a single context string
4. **Prompt Creation**: Context and question are formatted into a prompt
5. **LLM Generation**: LangChain wrapper calls Google Gemini API to generate answer
6. **Response Parsing**: Answer is extracted from LLM response
7. **Source Citation**: Source documents are included in response

## Dependencies

- `langchain_google_genai` - For ChatGoogleGenerativeAI wrapper
- `langchain_core` - For prompts and document types
- `google-generativeai` - Used by LangChain wrapper (indirect dependency)

## Error Handling

- Automatic retry logic (built into LangChain wrapper)
- Error messages for missing API keys
- Graceful handling of empty retrieval results
- Clear error messages for API failures

## Performance Considerations

- **Retrieval**: Fast (vector similarity search in ChromaDB)
- **Generation**: Depends on LLM response time (typically 2-5 seconds)
- **Context Size**: Limited by retrieved documents (default: 5 chunks)
- **Token Limits**: Max output tokens set to 2048

## Future Enhancements

- Support for multiple models
- Streaming responses
- Conversation history
- Multi-turn questions
- Advanced retrieval strategies (reranking, filtering)

