"""
FastAPI Backend for LangGraph Multi-Agent Research System
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from .langgraph_multiagent import run_research_analysis, create_research_graph
    from .document_processor import DocumentProcessor
except ImportError:
    # For script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from langgraph_multiagent import run_research_analysis, create_research_graph
    from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Research System API",
    description="API for running multi-agent research analysis on documents",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for research results (use Redis/DB in production)
research_results: Dict[str, Dict] = {}
research_status: Dict[str, str] = {}

# Document processing status
document_processing_status: Dict[str, str] = {}
document_processing_results: Dict[str, Dict] = {}

# Documents folder path
DOCUMENTS_FOLDER = Path("uploaded_documents")
DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)


# --- PYDANTIC MODELS ---
class ResearchRequest(BaseModel):
    """Request model for starting research analysis"""
    thread_id: Optional[str] = None
    description: Optional[str] = "Research analysis on uploaded documents"


class ResearchStatusResponse(BaseModel):
    """Response model for research status"""
    thread_id: str
    status: str
    current_step: int
    message: str


class CollaborationLogEntry(BaseModel):
    """Schema for collaboration log entry"""
    agent: str
    step: int
    action: str
    input: Dict
    output: Dict
    next_agent: Optional[str] = None


class ResearchResultResponse(BaseModel):
    """Response model for complete research results"""
    thread_id: str
    status: str
    workflow_status: str
    current_step: int

    # Researcher outputs
    researcher_analysis: Optional[str] = None
    researcher_findings: Optional[List[Dict]] = None
    researcher_sources: Optional[List[Dict]] = None
    researcher_status: Optional[str] = None

    # Reviewer outputs
    reviewer_critique: Optional[str] = None
    reviewer_strengths: Optional[List[str]] = None
    reviewer_weaknesses: Optional[List[str]] = None
    reviewer_status: Optional[str] = None

    # Synthesizer outputs
    synthesizer_synthesis: Optional[str] = None
    synthesizer_hypotheses: Optional[List[Dict]] = None
    synthesizer_status: Optional[str] = None

    # Questioner outputs
    questioner_gap_analysis: Optional[str] = None
    questioner_questions: Optional[List[str]] = None
    questioner_status: Optional[str] = None

    # Formatter outputs
    final_report: Optional[str] = None
    formatter_status: Optional[str] = None

    # Collaboration tracking
    collaboration_log: Optional[List[Dict]] = None

    error_message: Optional[str] = None
    timestamp: Optional[str] = None


class CollaborationResponse(BaseModel):
    """Response model for collaboration log"""
    thread_id: str
    collaboration_log: List[Dict]
    total_steps: int




# --- HELPER FUNCTIONS ---
def convert_state_to_json(state: Dict) -> Dict:
    """Convert LangGraph state to JSON-serializable format"""
    result = {
        "workflow_status": state.get("workflow_status", "unknown"),
        "current_step": state.get("current_step", 0),
        "error_message": state.get("error_message", ""),

        # Researcher
        "researcher_analysis": state.get("researcher_analysis", ""),
        "researcher_findings": state.get("researcher_findings", []),
        "researcher_sources": state.get("researcher_sources", []),
        "researcher_status": state.get("researcher_status", ""),

        # Reviewer
        "reviewer_critique": state.get("reviewer_critique", ""),
        "reviewer_strengths": state.get("reviewer_strengths", []),
        "reviewer_weaknesses": state.get("reviewer_weaknesses", []),
        "reviewer_status": state.get("reviewer_status", ""),

        # Synthesizer
        "synthesizer_synthesis": state.get("synthesizer_synthesis", ""),
        "synthesizer_hypotheses": state.get("synthesizer_hypotheses", []),
        "synthesizer_status": state.get("synthesizer_status", ""),

        # Questioner
        "questioner_gap_analysis": state.get("questioner_gap_analysis", ""),
        "questioner_questions": state.get("questioner_questions", []),
        "questioner_status": state.get("questioner_status", ""),

        # Formatter
        "final_report": state.get("final_report", ""),
        "formatter_status": state.get("formatter_status", ""),

        # Collaboration
        "collaboration_log": state.get("collaboration_log", []),

        "timestamp": datetime.now().isoformat()
    }

    return result


def process_documents_background(process_id: str, clear_existing: bool = True, generate_summaries: bool = True):
    """Process documents in background"""
    try:
        logger.info(f"Starting document processing for process_id: {process_id}")
        document_processing_status[process_id] = "running"
        
        # Initialize DocumentProcessor
        processor = DocumentProcessor(
            documents_folder=str(DOCUMENTS_FOLDER),
            vector_db_path="vector_db",
            summaries_db_path="summaries_vector_db",
            summaries_text_path="summaries"
        )
        
        # Process documents (clears vector DB if clear_existing=True)
        result = processor.process_documents(
            clear_existing=clear_existing,
            generate_summaries=generate_summaries
        )
        
        # Store results
        document_processing_results[process_id] = result
        document_processing_status[process_id] = result.get("status", "completed")
        
        logger.info(f"Document processing completed for process_id: {process_id}")
        logger.info(f"Processed {result.get('documents_processed', 0)} documents")
        logger.info(f"Created {result.get('chunks_created', 0)} chunks")
        logger.info(f"Generated {result.get('summaries_generated', 0)} summaries")
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}", exc_info=True)
        document_processing_status[process_id] = "error"
        document_processing_results[process_id] = {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


def run_research_background(thread_id: str):
    """Run research analysis in background with real-time updates"""
    try:
        logger.info(f"Starting research analysis for thread: {thread_id}")
        research_status[thread_id] = "running"
        
        # Initialize empty result structure
        research_results[thread_id] = convert_state_to_json({
            "workflow_status": "running",
            "current_step": 0,
            "error_message": "",
            "researcher_analysis": "",
            "researcher_findings": [],
            "researcher_sources": [],
            "researcher_status": "",
            "reviewer_critique": "",
            "reviewer_strengths": [],
            "reviewer_weaknesses": [],
            "reviewer_status": "",
            "synthesizer_synthesis": "",
            "synthesizer_hypotheses": [],
            "synthesizer_status": "",
            "questioner_gap_analysis": "",
            "questioner_questions": [],
            "questioner_status": "",
            "final_report": "",
            "formatter_status": "",
            "collaboration_log": []
        })

        config = {"configurable": {"thread_id": thread_id}}
        
        # Use streaming to get intermediate updates
        from langgraph_multiagent import create_research_graph
        graph = create_research_graph()
        
        initial_state = {
            "messages": [],
            "workflow_status": "started",
            "current_step": 0,
            "researcher_status": "",
            "reviewer_status": "",
            "synthesizer_status": "",
            "questioner_status": "",
            "formatter_status": "",
            "error_message": "",
            "collaboration_log": []
        }
        
        # Stream the graph execution to get real-time updates
        logger.info(f"Streaming research analysis for thread: {thread_id}")
        final_state = None
        
        for event in graph.stream(initial_state, config):
            # Update results with intermediate state
            for node_name, node_state in event.items():
                if node_name != "__end__":
                    # Store final state
                    final_state = node_state
                    
                    # Convert current state to JSON and update results
                    json_state = convert_state_to_json(node_state)
                    research_results[thread_id].update(json_state)
                    research_status[thread_id] = json_state.get("workflow_status", "running")
                    
                    # Log progress
                    current_step = json_state.get("current_step", 0)
                    workflow_status = json_state.get("workflow_status", "running")
                    logger.info(f"Thread {thread_id}: Step {current_step} completed by {node_name}, Status: {workflow_status}")
        
        # Store final result if we have it
        if final_state:
            json_result = convert_state_to_json(final_state)
            research_results[thread_id] = json_result
            research_status[thread_id] = json_result["workflow_status"]
        else:
            # Fallback: if streaming didn't work, use invoke
            logger.warning(f"Streaming returned no final state, using invoke for thread: {thread_id}")
            final_state = graph.invoke(initial_state, config)
            json_result = convert_state_to_json(final_state)
            research_results[thread_id] = json_result
            research_status[thread_id] = json_result["workflow_status"]

        logger.info(f"Research analysis completed for thread: {thread_id}")

        # Save to file
        output_file = f"research_output_{thread_id}.json"
        with open(output_file, "w") as f:
            json.dump(json_result, f, indent=2)
        logger.info(f"Saved results to {output_file}")

    except Exception as e:
        logger.error(f"Error in research analysis: {str(e)}", exc_info=True)
        research_status[thread_id] = "error"
        research_results[thread_id] = {
            "workflow_status": "error",
            "error_message": str(e),
            "timestamp": datetime.now().isoformat()
        }


# --- API ENDPOINTS ---
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent Research System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# --- DOCUMENT UPLOAD AND PROCESSING ENDPOINTS ---
@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload research papers to the uploaded_documents folder.
    Returns list of uploaded file names.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Supported file extensions
    supported_extensions = {'.pdf', '.txt', '.docx', '.doc'}
    uploaded_files = []
    errors = []
    
    for file in files:
        try:
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in supported_extensions:
                errors.append(f"{file.filename}: Unsupported file type. Supported types: PDF, TXT, DOCX, DOC")
                continue
            
            # Save file to uploaded_documents folder
            file_path = DOCUMENTS_FOLDER / file.filename
            
            # If file already exists, add timestamp to avoid overwriting
            if file_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = file_path.stem
                file_path = DOCUMENTS_FOLDER / f"{stem}_{timestamp}{file_ext}"
            
            # Read file content and save
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            uploaded_files.append(file_path.name)
            logger.info(f"Uploaded file: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {str(e)}")
            errors.append(f"{file.filename}: {str(e)}")
    
    if not uploaded_files and errors:
        raise HTTPException(status_code=400, detail=f"Failed to upload files: {errors}")
    
    return {
        "status": "success",
        "uploaded_files": uploaded_files,
        "errors": errors if errors else None,
        "message": f"Successfully uploaded {len(uploaded_files)} file(s)"
    }


@app.post("/api/documents/process")
async def process_documents(
    background_tasks: BackgroundTasks,
    clear_existing: bool = True,
    generate_summaries: bool = True
):
    """
    Process all documents in the uploaded_documents folder.
    This will:
    1. Clear the vector database (if clear_existing=True)
    2. Load documents using DocumentLoader
    3. Process documents using DocumentProcessor
    4. Generate embeddings and summaries
    
    The processing runs in the background.
    """
    
    # Generate process_id
    process_id = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if documents folder has any files
    files = list(DOCUMENTS_FOLDER.glob("*"))
    supported_files = [f for f in files if f.suffix.lower() in {'.pdf', '.txt', '.docx', '.doc'}]
    
    if not supported_files:
        raise HTTPException(
            status_code=400,
            detail="No documents found in uploaded_documents folder. Please upload documents first."
        )
    
    # Start background processing
    background_tasks.add_task(
        process_documents_background,
        process_id,
        clear_existing,
        generate_summaries
    )
    
    document_processing_status[process_id] = "started"
    
    return {
        "status": "started",
        "process_id": process_id,
        "message": f"Document processing started. Processing {len(supported_files)} file(s).",
        "clear_existing": clear_existing,
        "generate_summaries": generate_summaries
    }


@app.get("/api/documents/process/status/{process_id}")
async def get_process_status(process_id: str):
    """Get the status of document processing"""
    if process_id not in document_processing_status:
        raise HTTPException(status_code=404, detail="Process ID not found")
    
    status = document_processing_status[process_id]
    result = document_processing_results.get(process_id, {})
    
    return {
        "process_id": process_id,
        "status": status,
        "result": result
    }


@app.get("/api/documents/list")
async def list_documents():
    """List all documents in the uploaded_documents folder"""
    try:
        files = list(DOCUMENTS_FOLDER.glob("*"))
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        
        documents = []
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return {
            "status": "success",
            "total": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the uploaded_documents folder"""
    try:
        file_path = DOCUMENTS_FOLDER / filename
        
        # Security check: ensure file is in the documents folder
        if not file_path.resolve().is_relative_to(DOCUMENTS_FOLDER.resolve()):
            raise HTTPException(status_code=403, detail="Invalid file path")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        logger.info(f"Deleted file: {filename}")
        
        return {
            "status": "success",
            "message": f"File {filename} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/start", response_model=ResearchStatusResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new research analysis.
    The analysis runs in the background and results can be fetched later.
    """
    # Generate thread_id if not provided
    thread_id = request.thread_id or f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Check if already running
    if thread_id in research_status and research_status[thread_id] == "running":
        raise HTTPException(status_code=400, detail="Research already running for this thread_id")

    # Start background task
    background_tasks.add_task(run_research_background, thread_id)

    research_status[thread_id] = "started"

    return ResearchStatusResponse(
        thread_id=thread_id,
        status="started",
        current_step=0,
        message=f"Research analysis started for thread: {thread_id}"
    )


@app.get("/api/research/status/{thread_id}", response_model=ResearchStatusResponse)
async def get_research_status(thread_id: str):
    """
    Get the current status of a research analysis.
    """
    if thread_id not in research_status:
        raise HTTPException(status_code=404, detail="Research thread not found")

    status = research_status[thread_id]
    current_step = 0

    if thread_id in research_results:
        current_step = research_results[thread_id].get("current_step", 0)

    return ResearchStatusResponse(
        thread_id=thread_id,
        status=status,
        current_step=current_step,
        message=f"Research status: {status}"
    )


@app.get("/api/research/result/{thread_id}", response_model=ResearchResultResponse)
async def get_research_result(thread_id: str):
    """
    Get the research results for a thread.
    Returns partial results if still running, complete results when done.
    Returns 404 if not found.
    """
    if thread_id not in research_status:
        raise HTTPException(status_code=404, detail="Research thread not found")

    status = research_status[thread_id]

    if thread_id not in research_results:
        raise HTTPException(status_code=404, detail="Research results not available")

    result = research_results[thread_id]

    # Return partial results even if still running (for real-time updates)
    return ResearchResultResponse(
        thread_id=thread_id,
        status=status,
        **result
    )


@app.get("/api/research/collaboration/{thread_id}", response_model=CollaborationResponse)
async def get_collaboration_log(thread_id: str):
    """
    Get the agent collaboration log for a specific research thread.
    This shows how agents interacted and built upon each other's work.
    """
    if thread_id not in research_status:
        raise HTTPException(status_code=404, detail="Research thread not found")

    if thread_id not in research_results:
        raise HTTPException(status_code=404, detail="Research results not available")

    result = research_results[thread_id]
    collaboration_log = result.get("collaboration_log", [])

    return CollaborationResponse(
        thread_id=thread_id,
        collaboration_log=collaboration_log,
        total_steps=len(collaboration_log)
    )


@app.get("/api/research/list")
async def list_research_threads():
    """
    List all research threads and their status.
    """
    threads = []
    for thread_id, status in research_status.items():
        thread_info = {
            "thread_id": thread_id,
            "status": status,
            "current_step": 0
        }
        if thread_id in research_results:
            thread_info["current_step"] = research_results[thread_id].get("current_step", 0)
            thread_info["timestamp"] = research_results[thread_id].get("timestamp")
        threads.append(thread_info)

    return {
        "total": len(threads),
        "threads": threads
    }


@app.delete("/api/research/{thread_id}")
async def delete_research(thread_id: str):
    """
    Delete a research thread and its results.
    """
    if thread_id not in research_status:
        raise HTTPException(status_code=404, detail="Research thread not found")

    # Remove from memory
    if thread_id in research_status:
        del research_status[thread_id]
    if thread_id in research_results:
        del research_results[thread_id]

    # Remove file if exists
    output_file = f"research_output_{thread_id}.json"
    if os.path.exists(output_file):
        os.remove(output_file)

    return {
        "message": f"Research thread {thread_id} deleted successfully"
    }


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "src.api_server:app",  # Use string format for better compatibility
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for stability
        log_level="info"
    )