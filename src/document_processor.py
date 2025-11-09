"""
Main Document Processor
Orchestrates the document loading and vector store pipeline.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from document_loader import DocumentLoader
from vector_store import VectorStore
from summarizer_agent import SummarizerAgent

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

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
    - SummarizerAgent: Generates summaries of documents

    You typically use this class instead of using DocumentLoader and VectorStore directly.
    """

    def __init__(
        self,
        documents_folder: str = "uploaded_documents",
        vector_db_path: str = "vector_db",
        summaries_db_path: str = "summaries_vector_db",
        summaries_text_path: str = "summaries",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document processor.

        This creates components:
        1. DocumentLoader - handles file reading and chunking
        2. VectorStore - handles embedding storage and search for documents
        3. VectorStore - separate instance for summaries
        4. SummarizerAgent - generates document summaries

        Args:
            documents_folder: Path to folder containing documents
            vector_db_path: Path to vector database storage for documents
            summaries_db_path: Path to vector database storage for summaries
            summaries_text_path: Path to store summary text files
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.documents_folder = documents_folder
        self.vector_db_path = vector_db_path
        self.summaries_db_path = summaries_db_path
        self.summaries_text_path = Path(summaries_text_path)

        # Create summaries directory
        self.summaries_text_path.mkdir(parents=True, exist_ok=True)

        # Initialize DocumentLoader (reads files, creates chunks)
        logger.info("Initializing document loader...")
        self.document_loader = DocumentLoader(
            documents_folder=documents_folder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Initialize VectorStore for documents (stores embeddings, enables search)
        logger.info("Initializing document vector store...")
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            collection_name="research_documents"
        )

        # Initialize VectorStore for summaries
        logger.info("Initializing summaries vector store...")
        self.summaries_vector_store = VectorStore(
            persist_directory=summaries_db_path,
            collection_name="document_summaries"
        )

        # Initialize SummarizerAgent
        logger.info("Initializing summarizer agent...")
        self.summarizer = SummarizerAgent()

    def _format_summary_as_text(self, summary: Dict[str, Any]) -> str:
        """
        Format a summary dictionary as readable text.

        Args:
            summary: Summary dictionary

        Returns:
            Formatted text string
        """
        text_parts = []

        text_parts.append(f"Title: {summary.get('title', 'N/A')}")
        text_parts.append(f"Authors: {summary.get('authors', 'N/A')}")
        text_parts.append(f"Year: {summary.get('year', 'N/A')}")
        text_parts.append(f"\nResearch Question:\n{summary.get('research_question', 'N/A')}")
        text_parts.append(f"\nMethods:\n{summary.get('methods', 'N/A')}")
        text_parts.append(f"\nKey Findings:\n{summary.get('key_findings', 'N/A')}")
        text_parts.append(f"\nLimitations:\n{summary.get('limitations', 'N/A')}")
        text_parts.append(f"\nSource File: {summary.get('source_file', 'N/A')}")

        return "\n".join(text_parts)

    def _save_summary_as_text(self, summary: Dict[str, Any], file_path: str):
        """
        Save a summary as a text file.

        Args:
            summary: Summary dictionary
            file_path: Original file path
        """
        try:
            # Generate filename from source file
            source_file = Path(file_path)
            summary_filename = f"{source_file.stem}_summary.txt"
            summary_path = self.summaries_text_path / summary_filename

            # Format summary as text
            summary_text = self._format_summary_as_text(summary)

            # Save to file
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)

            logger.info(f"Saved summary to: {summary_path}")

        except Exception as e:
            logger.error(f"Error saving summary as text: {str(e)}")

    def _save_summary_as_json(self, summary: Dict[str, Any], file_path: str):
        """
        Save a summary as a JSON file.

        Args:
            summary: Summary dictionary
            file_path: Original file path
        """
        try:
            # Generate filename from source file
            source_file = Path(file_path)
            summary_filename = f"{source_file.stem}_summary.json"
            summary_path = self.summaries_text_path / summary_filename

            # Save to file
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Saved summary JSON to: {summary_path}")

        except Exception as e:
            logger.error(f"Error saving summary as JSON: {str(e)}")

    def process_documents(self, clear_existing: bool = False, generate_summaries: bool = True) -> dict:
        """
        Process all documents from the uploaded_documents folder.

        This method coordinates the complete pipeline:
        1. Uses DocumentLoader to read files and create chunks
        2. Uses VectorStore to store chunks as embeddings
        3. Optionally generates summaries and stores them in both text and vector form

        Args:
            clear_existing: If True, clear existing vector stores before processing
            generate_summaries: If True, generate and store summaries

        Returns:
            Dictionary with processing results
        """
        try:
            # Clear existing if requested
            if clear_existing:
                logger.warning("Clearing existing vector stores...")
                self.vector_store.delete_collection()
                self.vector_store = VectorStore(
                    persist_directory=self.vector_db_path,
                    collection_name="research_documents"
                )

                if generate_summaries:
                    self.summaries_vector_store.delete_collection()
                    self.summaries_vector_store = VectorStore(
                        persist_directory=self.summaries_db_path,
                        collection_name="document_summaries"
                    )

            # STEP 1: Use DocumentLoader to read files and create chunks
            logger.info("Loading documents...")
            if generate_summaries:
                chunks, full_texts = self.document_loader.process_all_with_full_text()
            else:
                chunks = self.document_loader.process_all()
                full_texts = {}

            if not chunks:
                logger.warning("No documents found to process")
                return {
                    "status": "error",
                    "message": "No documents found in uploaded_documents folder",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "summaries_generated": 0
                }

            # STEP 2: Use VectorStore to store chunks as embeddings
            logger.info(f"Adding {len(chunks)} chunks to document vector store...")
            document_ids = self.vector_store.add_documents(chunks)

            # Get collection info
            collection_info = self.vector_store.get_collection_info()

            # STEP 3: Generate and store summaries
            summaries_generated = 0
            summaries_stored = 0

            if generate_summaries and full_texts:
                logger.info(f"Generating summaries for {len(full_texts)} documents...")
                summary_documents = []

                for file_path, full_text in full_texts.items():
                    logger.info(f"Generating summary for: {file_path}")

                    # Generate summary
                    summary = self.summarizer.summarize_paper(full_text, file_path)

                    if summary.get('summary_generated', False):
                        summaries_generated += 1

                        # Save summary as text file
                        self._save_summary_as_text(summary, file_path)

                        # Save summary as JSON file
                        self._save_summary_as_json(summary, file_path)

                        # Create Document object for vector storage
                        summary_text = self._format_summary_as_text(summary)
                        summary_doc = Document(
                            page_content=summary_text,
                            metadata={
                                "source_file": file_path,
                                "title": summary.get('title', ''),
                                "authors": summary.get('authors', ''),
                                "year": summary.get('year', ''),
                                "document_type": "summary"
                            }
                        )
                        summary_documents.append(summary_doc)
                    else:
                        logger.warning(f"Failed to generate summary for: {file_path}")

                # Store summaries in vector database
                if summary_documents:
                    logger.info(f"Adding {len(summary_documents)} summaries to vector store...")
                    self.summaries_vector_store.add_documents(summary_documents)
                    summaries_stored = len(summary_documents)

            # Get summaries collection info
            summaries_collection_info = self.summaries_vector_store.get_collection_info() if generate_summaries else {}

            result = {
                "status": "success",
                "documents_processed": len(chunks),
                "chunks_created": len(chunks),
                "document_ids": len(document_ids),
                "collection_info": collection_info,
                "summaries_generated": summaries_generated,
                "summaries_stored": summaries_stored,
                "summaries_collection_info": summaries_collection_info,
                "summaries_text_path": str(self.summaries_text_path)
            }

            logger.info("Document processing completed successfully!")
            logger.info(f"Processed {len(chunks)} document chunks")
            logger.info(f"Generated {summaries_generated} summaries")
            logger.info(f"Vector store contains {collection_info.get('document_count', 0)} documents")
            logger.info(f"Summaries vector store contains {summaries_collection_info.get('document_count', 0)} summaries")

            return result

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "documents_processed": 0,
                "chunks_created": 0,
                "summaries_generated": 0
            }

    def search_documents(self, query: str, k: int = 4) -> List:
        """
        Search for similar documents in the main vector store.

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

    def search_summaries(self, query: str, k: int = 4) -> List:
        """
        Search for similar summaries in the summaries vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar summaries
        """
        try:
            results = self.summaries_vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching summaries: {str(e)}")
            return []

    def get_stats(self) -> dict:
        """Get statistics about both vector stores."""
        try:
            document_stats = self.vector_store.get_collection_info()
            summary_stats = self.summaries_vector_store.get_collection_info()

            return {
                "documents": document_stats,
                "summaries": summary_stats,
                "summaries_text_path": str(self.summaries_text_path)
            }
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
        help="Clear existing vector stores before processing"
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
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip summary generation"
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
    result = processor.process_documents(
        clear_existing=args.clear,
        generate_summaries=not args.no_summaries
    )

    # Print results
    if result["status"] == "success":
        print("\n" + "="*50)
        print("Document Processing Complete!")
        print("="*50)
        print(f"Documents processed: {result['documents_processed']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Summaries generated: {result['summaries_generated']}")
        print(f"Summaries stored in vector DB: {result['summaries_stored']}")
        print(f"Collection info: {result['collection_info']}")
        print(f"Summaries collection info: {result['summaries_collection_info']}")
        print(f"Summaries text files saved to: {result['summaries_text_path']}")
        print("="*50 + "\n")
    else:
        print(f"\nError: {result['message']}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()