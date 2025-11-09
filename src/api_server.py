"""
FastAPI Backend for LangGraph Multi-Agent Research System
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from .langgraph_multiagent import run_research_analysis, create_research_graph
except ImportError:
    # For script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from langgraph_multiagent import run_research_analysis, create_research_graph

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


def run_research_background(thread_id: str):
    """Run research analysis in background"""
    try:
        logger.info(f"Starting research analysis for thread: {thread_id}")
        research_status[thread_id] = "running"

        config = {"configurable": {"thread_id": thread_id}}
        result = run_research_analysis(config)

        # Convert to JSON-serializable format
        json_result = convert_state_to_json(result)

        # Store result
        research_results[thread_id] = json_result
        research_status[thread_id] = json_result["workflow_status"]

        logger.info(f"Research analysis completed for thread: {thread_id}")

        # Save to file
        output_file = f"research_output_{thread_id}.json"
        with open(output_file, "w") as f:
            json.dump(json_result, f, indent=2)
        logger.info(f"Saved results to {output_file}")

    except Exception as e:
        logger.error(f"Error in research analysis: {str(e)}")
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
    Get the complete research results for a thread.
    Returns 404 if not found, 202 if still running.
    """
    if thread_id not in research_status:
        raise HTTPException(status_code=404, detail="Research thread not found")

    status = research_status[thread_id]

    if status == "running" or status == "started":
        raise HTTPException(status_code=202, detail="Research still in progress")

    if thread_id not in research_results:
        raise HTTPException(status_code=404, detail="Research results not available")

    result = research_results[thread_id]

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
        "fastapi_research_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )