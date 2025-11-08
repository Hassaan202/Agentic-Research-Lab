"""
RAG Pipeline Module
Implements Retrieval-Augmented Generation for answering questions using document context.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
import logging
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Handle both script execution and module import
try:
    from .vector_store import VectorStore
except ImportError:
    # For script execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline for answering questions using document retrieval and LLM generation.
    
    This class combines:
    - VectorStore: Retrieves relevant document chunks
    - ChatGoogleGenerativeAI: Generates answers based on retrieved context
    """
    
    def __init__(
        self,
        vector_db_path: str = "vector_db",
        collection_name: str = "research_documents",
        llm_model: str = "gemini-2.5-flash",  
        temperature: float = 0.7,
        max_retrieval_docs: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_db_path: Path to the vector database
            collection_name: Name of the ChromaDB collection
            llm_model: Google Gemini model name (will be overridden to "gemini-2.5-flash" - the only working model)
            temperature: LLM temperature for response generation
            max_retrieval_docs: Maximum number of documents to retrieve for context
        """
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retrieval_docs = max_retrieval_docs
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please create a .env file with your Google API key. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Initialize VectorStore for retrieval
        logger.info("Initializing vector store...")
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name
        )
        
        # Initialize LLM
        # Using only the working configuration: gemini-2.5-flash with LangChain wrapper
        logger.info(f"Initializing LLM: {llm_model}...")
        
        # Force use of working model: gemini-2.5-flash
        # This is the only model that works reliably with LangChain wrapper
        if llm_model != "gemini-2.5-flash":
            logger.warning(f"Requested model '{llm_model}' will be overridden to 'gemini-2.5-flash' (working model)")
            llm_model = "gemini-2.5-flash"
        
        # Initialize with LangChain wrapper (this is the working configuration)
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
        # Store model information
        self.actual_model_name = llm_model
        self.api_method = "LangChain wrapper"
        self.use_direct_api = False
        self.genai_model = None
        self.temperature = temperature
        
        logger.info(f"LLM initialized successfully with model: {llm_model} (LangChain wrapper)")
        print(f"✓ Using model: {llm_model} (via LangChain wrapper)")
        
        logger.info("RAG pipeline initialized successfully")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for RAG."""
        template = """You are a helpful research assistant. Answer the user's question based on the following context from research documents.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear and concise answer based on the context provided
- If the context doesn't contain enough information to answer the question, say so
- Cite specific details from the context when relevant
- If multiple sources discuss the topic, synthesize the information
- Use proper formatting and structure in your response

Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def answer_question(
        self,
        question: str,
        k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question
            k: Number of documents to retrieve (defaults to max_retrieval_docs)
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        if k is None:
            k = self.max_retrieval_docs
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving relevant documents for question: {question[:50]}...")
            retrieved_docs = self.vector_store.similarity_search(
                query=question,
                k=k
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant documents to answer your question. Please make sure the vector store has been populated with documents.",
                    "sources": [],
                    "question": question
                }
            
            # Step 2: Combine retrieved documents into context
            context = "\n\n".join([
                f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Step 3: Create prompt with context and question
            prompt_template = self._create_prompt_template()
            prompt_text = prompt_template.format(context=context, question=question)
            
            # Step 4: Generate answer using LLM (LangChain wrapper)
            logger.info("Generating answer using LLM...")
            
            # Use LangChain wrapper (only working method)
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt_text)]
            response = self.llm.invoke(messages)
            
            # Extract answer from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            result = {
                "answer": answer,
                "question": question,
                "num_sources": len(retrieved_docs)
            }
            
            if return_sources:
                result["sources"] = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", None),
                        "chunk_index": doc.metadata.get("chunk_index", None),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    for doc in retrieved_docs
                ]
            
            logger.info("Answer generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "question": question,
                "sources": [],
                "error": str(e)
            }
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return self.vector_store.get_collection_info()
    
    def get_model_info(self) -> Dict:
        """Get information about the current LLM model."""
        return {
            "model_name": self.actual_model_name,
            "api_method": self.api_method,
            "temperature": self.temperature,
            "use_direct_api": self.use_direct_api
        }


def main():
    """Interactive Q&A interface for the RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline - Ask questions about your documents")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Google Gemini model (only gemini-2.5-flash is supported and will be used)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (0.0 to 1.0)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        default="vector_db",
        help="Path to vector database"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY not found. Please create a .env file with your Google API key. "
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
        sys.exit(1)
    
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline(
            vector_db_path=args.vector_db,
            llm_model=args.model,
            temperature=args.temperature,
            max_retrieval_docs=args.k
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        sys.exit(1)
    
    # Get vector store stats
    stats = rag.get_vector_store_stats()
    print("\n" + "="*60)
    print("RAG Pipeline - Research Document Q&A")
    print("="*60)
    print(f"Vector Store: {stats.get('document_count', 0)} documents")
    print(f"Requested Model: {args.model}")
    print(f"Active Model: {rag.actual_model_name}")
    print(f"API Method: {rag.api_method}")
    print(f"Temperature: {args.temperature}")
    print(f"Retrieval Docs (k): {args.k}")
    print("="*60)
    print(f"\n💡 Tip: To use only this model, set --model {rag.actual_model_name}\n")
    
    # Single question mode
    if args.question:
        result = rag.answer_question(args.question, k=args.k)
        print(f"\nQuestion: {result['question']}\n")
        print(f"Answer:\n{result['answer']}\n")
        if result.get('sources'):
            print(f"\nSources ({result['num_sources']} documents):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source['source']}")
                if source.get('page'):
                    print(f"   Page: {source['page']}")
                print(f"   Preview: {source['content_preview']}")
        return
    
    # Interactive mode
    print("Enter your questions (type 'quit' or 'exit' to stop, 'help' for commands)\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'help':
                print("\nCommands:")
                print("  help     - Show this help message")
                print("  stats    - Show vector store and model statistics")
                print("  model    - Show current model information")
                print("  quit/exit - Exit the program")
                print("  Any other text - Ask a question\n")
                continue
            
            if question.lower() == 'stats':
                stats = rag.get_vector_store_stats()
                model_info = rag.get_model_info()
                print(f"\nVector Store Statistics:")
                print(f"  Documents: {stats.get('document_count', 0)}")
                print(f"  Collection: {stats.get('collection_name', 'N/A')}")
                print(f"  Path: {stats.get('persist_directory', 'N/A')}")
                print(f"\nModel Information:")
                print(f"  Active Model: {model_info.get('model_name', 'N/A')}")
                print(f"  API Method: {model_info.get('api_method', 'N/A')}")
                print(f"  Temperature: {model_info.get('temperature', 'N/A')}")
                print(f"  Direct API: {model_info.get('use_direct_api', False)}\n")
                continue
            
            if question.lower() == 'model':
                model_info = rag.get_model_info()
                print(f"\nCurrent Model Configuration:")
                print(f"  Model Name: {model_info.get('model_name', 'N/A')}")
                print(f"  API Method: {model_info.get('api_method', 'N/A')}")
                print(f"  Temperature: {model_info.get('temperature', 'N/A')}")
                print(f"\nTo use only this model, run:")
                print(f"  python src/rag_pipeline.py --model {model_info.get('model_name', 'N/A')}\n")
                continue
            
            # Answer the question
            print("\nProcessing...")
            result = rag.answer_question(question, k=args.k)
            
            print(f"\n{'='*60}")
            print(f"Answer:")
            print(f"{'='*60}")
            print(result['answer'])
            print(f"{'='*60}")
            
            if result.get('sources'):
                print(f"\nSources ({result['num_sources']} documents):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{i}. {source['source']}")
                    if source.get('page'):
                        print(f"   Page: {source['page']}")
                    if source.get('chunk_index') is not None:
                        print(f"   Chunk: {source['chunk_index']}")
                    print(f"   Preview: {source['content_preview']}")
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()

