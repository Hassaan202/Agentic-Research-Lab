"""
Main Document Processor
Orchestrates the document loading and vector store pipeline.
"""

import os
import sys
from pathlib import Path
from typing import List
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from document_loader import DocumentLoader
from vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main processor for ingesting documents and storing embeddings.
    
    This class ORCHESTRATES the document processing pipeline by using:
    - DocumentLoader: Reads files and creates text chunks
    - VectorStore: Stores chunks as embeddings in the database
    
    You typically use this class instead of using DocumentLoader and VectorStore directly.
    """
    
    def __init__(
        self,
        documents_folder: str = "uploaded_documents",
        vector_db_path: str = "vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document processor.
        
        This creates two components:
        1. DocumentLoader - handles file reading and chunking
        2. VectorStore - handles embedding storage and search
        
        Args:
            documents_folder: Path to folder containing documents
            vector_db_path: Path to vector database storage
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.documents_folder = documents_folder
        self.vector_db_path = vector_db_path
        
        # Initialize DocumentLoader (reads files, creates chunks)
        logger.info("Initializing document loader...")
        self.document_loader = DocumentLoader(
            documents_folder=documents_folder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize VectorStore (stores embeddings, enables search)
        logger.info("Initializing vector store...")
        self.vector_store = VectorStore(persist_directory=vector_db_path)
    
    def process_documents(self, clear_existing: bool = False) -> dict:
        """
        Process all documents from the uploaded_documents folder.
        
        This method coordinates the complete pipeline:
        1. Uses DocumentLoader to read files and create chunks
        2. Uses VectorStore to store chunks as embeddings
        
        Args:
            clear_existing: If True, clear existing vector store before processing
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Clear existing if requested
            if clear_existing:
                logger.warning("Clearing existing vector store...")
                self.vector_store.delete_collection()
                self.vector_store = VectorStore(persist_directory=self.vector_db_path)
            
            # STEP 1: Use DocumentLoader to read files and create chunks
            # DocumentLoader reads PDF/TXT files and splits them into text chunks
            logger.info("Loading documents...")
            chunks = self.document_loader.process_all()
            
            if not chunks:
                logger.warning("No documents found to process")
                return {
                    "status": "error",
                    "message": "No documents found in uploaded_documents folder",
                    "documents_processed": 0,
                    "chunks_created": 0
                }
            
            # STEP 2: Use VectorStore to store chunks as embeddings
            # VectorStore converts text chunks to embeddings and stores in ChromaDB
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            document_ids = self.vector_store.add_documents(chunks)
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info()
            
            result = {
                "status": "success",
                "documents_processed": len(chunks),
                "chunks_created": len(chunks),
                "document_ids": len(document_ids),
                "collection_info": collection_info
            }
            
            logger.info("Document processing completed successfully!")
            logger.info(f"Processed {len(chunks)} document chunks")
            logger.info(f"Vector store contains {collection_info.get('document_count', 0)} documents")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "documents_processed": 0,
                "chunks_created": 0
            }
    
    def search_documents(self, query: str, k: int = 4) -> List:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        try:
            return self.vector_store.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}


def main():
    """Main function to run the document processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents and store embeddings")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before processing"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="uploaded_documents",
        help="Path to documents folder"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for document splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for document splitting"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY not found. Please create a .env file with your Google API key. "
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
        sys.exit(1)
    
    # Initialize processor
    processor = DocumentProcessor(
        documents_folder=args.folder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process documents
    result = processor.process_documents(clear_existing=args.clear)
    
    # Print results
    if result["status"] == "success":
        print("\n" + "="*50)
        print("Document Processing Complete!")
        print("="*50)
        print(f"Documents processed: {result['documents_processed']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Collection info: {result['collection_info']}")
        print("="*50 + "\n")
    else:
        print(f"\nError: {result['message']}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

