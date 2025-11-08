"""
Document Loader Module
Handles loading and parsing of various document types from the uploaded_documents folder.
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads and processes documents from the uploaded_documents folder.
    
    RESPONSIBILITY: File reading and text chunking ONLY
    - Reads PDF, TXT, DOCX files
    - Extracts text from files
    - Splits text into chunks
    
    DOES NOT:
    - Create embeddings
    - Store in database
    - Handle vector search
    
    This is used by DocumentProcessor to get text chunks.
    """
    
    def __init__(
        self,
        documents_folder: str = "uploaded_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document loader.
        
        Args:
            documents_folder: Path to folder containing documents
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.documents_folder = Path(documents_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Supported file extensions
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
        }
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load a PDF file."""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            # Add metadata about source file
            for doc in documents:
                doc.metadata['source_file'] = str(file_path)
                doc.metadata['file_type'] = 'pdf'
            logger.info(f"Loaded PDF: {file_path.name} ({len(documents)} pages)")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """Load a text file."""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = str(file_path)
                doc.metadata['file_type'] = 'text'
            logger.info(f"Loaded text file: {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []
    
    def _load_docx(self, file_path: Path) -> List[Document]:
        """Load a Word document."""
        try:
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = str(file_path)
                doc.metadata['file_type'] = 'docx'
            logger.info(f"Loaded Word document: {file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading Word document {file_path}: {str(e)}")
            return []
    
    def load_documents(self) -> List[Document]:
        """
        Load all documents from the uploaded_documents folder.
        
        Returns:
            List of Document objects
        """
        if not self.documents_folder.exists():
            logger.warning(f"Documents folder not found: {self.documents_folder}")
            self.documents_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created documents folder: {self.documents_folder}")
            return []
        
        all_documents = []
        files_processed = 0
        
        # Get all supported files
        for ext in self.supported_extensions.keys():
            files = list(self.documents_folder.glob(f"*{ext}"))
            for file_path in files:
                loader_func = self.supported_extensions[ext]
                documents = loader_func(file_path)
                all_documents.extend(documents)
                files_processed += 1
        
        logger.info(f"Loaded {files_processed} files, {len(all_documents)} document pages")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            logger.warning("No documents to split")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            if 'source_file' not in chunk.metadata:
                chunk.metadata['source_file'] = 'unknown'
        
        return chunks
    
    def process_all(self) -> List[Document]:
        """
        Load and split all documents in one step.
        
        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        chunks = self.split_documents(documents)
        return chunks

