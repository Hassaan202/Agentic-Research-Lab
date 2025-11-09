"""
Vector Store Module
Manages the ChromaDB vector database for storing document embeddings.
"""

import os
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages document embeddings in ChromaDB vector database.
    
    RESPONSIBILITY: Embedding storage and search ONLY
    - Converts text chunks to embeddings (using Google Gemini)
    - Stores embeddings in ChromaDB
    - Searches for similar documents
    
    DOES NOT:
    - Read files from disk
    - Parse PDFs or other formats
    - Split text into chunks
    
    This is used by DocumentProcessor to store and search documents.
    """
    
    def __init__(
        self,
        persist_directory: str = "vector_db",
        collection_name: str = "research_documents",
        embedding_model: str = "models/text-embedding-004"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the ChromaDB collection
            embedding_model: Google Gemini embedding model name
                           Default: "models/text-embedding-004" (Latest Gemini embedding model)
                           Alternative: "models/embedding-001"
                           Note: Gemini 2.5 Flash is a language model, not an embedding model.
                                 We use Google's embedding models for vector search.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please create a .env file with your Google API key. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Initialize Google Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key
        )
        
        # Initialize ChromaDB
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load the ChromaDB vector store."""
        try:
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Check if vector store already exists
            if self._vector_store_exists():
                logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            else:
                logger.info(f"Creating new vector store at {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def _vector_store_exists(self) -> bool:
        """Check if vector store already exists."""
        # Check if ChromaDB files exist
        chroma_files = list(self.persist_directory.glob("*.sqlite3"))
        return len(chroma_files) > 0
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store...")
            
            # Add documents in batches
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Note: Chroma 0.4.x automatically persists documents, no need to call persist()
            
            logger.info(f"Successfully added {len(all_ids)} documents to vector store")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar Document objects
        """
        try:
            if filter:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=filter
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (Document, score)
        """
        try:
            if filter:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            raise
    
    def get_collection_info(self) -> dict:
        """Get information about the vector store collection."""
        try:
            count = self.vector_store._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)."""
        try:
            import shutil
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
                logger.warning(f"Deleted vector store at {self.persist_directory}")
                self._initialize_vector_store()
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
