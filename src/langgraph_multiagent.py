"""
Enhanced LangGraph Multi-Agent Research System
"""

import os
import logging
from typing import TypedDict, List, Annotated, Sequence, Dict, Optional
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from .rag_pipeline import RAGPipeline
except ImportError:
    # For script execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from rag_pipeline import RAGPipeline


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

memory = MemorySaver()

# Initialize RAG pipeline at module level
rag_pipeline = RAGPipeline(
    vector_db_path="vector_db",
    collection_name="research_documents"
)

summaries_rag_pipeline = RAGPipeline(
    vector_db_path="summaries_vector_db",
    collection_name="document_summaries"
)

# --- STATE SCHEMA ---
class ResearchState(TypedDict):
    """Simplified state focused on core outputs"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Researcher outputs
    researcher_analysis: str
    researcher_findings: List[Dict]
    researcher_sources: List[Dict]
    researcher_status: str

    # Reviewer outputs
    reviewer_critique: str
    reviewer_strengths: List[str]
    reviewer_weaknesses: List[str]
    reviewer_status: str

    # Synthesizer outputs
    synthesizer_synthesis: str
    synthesizer_hypotheses: List[Dict]
    synthesizer_status: str

    # Questioner outputs
    questioner_gap_analysis: str
    questioner_questions: List[str]
    questioner_status: str

    # Formatter outputs
    final_report: str
    formatter_status: str

    # Workflow metadata
    workflow_status: str
    error_message: str
    current_step: int




# ---AGENT NODES ---
def researcher_node(state: ResearchState) -> Dict:
    """
    RESEARCHER Agent: Analyzes research papers with citations.
    """
    logger.info("RESEARCHER: Starting analysis...")

    # Retrieve summaries for high-level overview
    logger.info("RESEARCHER: Retrieving document summaries...")
    summaries_result = summaries_rag_pipeline.answer_question(
        "Provide comprehensive summaries of all research papers including key findings, methodologies, and conclusions",
        k=10,
        return_sources=True
    )
    summaries_context = summaries_result['answer']
    summary_sources = summaries_result.get('sources', [])

    # Retrieve detailed context
    logger.info("RESEARCHER: Retrieving detailed document context...")
    context_result = rag_pipeline.answer_question(
        "Extract key findings, methodologies, results, and conclusions from all research documents",
        k=10,
        return_sources=True
    )
    context_text = context_result['answer']
    sources = context_result.get('sources', [])

    system_prompt = """You are a research analyst extracting key findings from academic papers.

YOUR TASK:
Analyze the provided research papers and extract 5-7 key findings with proper citations.

OUTPUT FORMAT (use exactly this structure):
Finding 1: [Clear finding statement]
Citation: [Source reference like [S1] or [D2]]
Evidence: [Specific quote or data from the paper]

Finding 2: [Clear finding statement]
Citation: [Source reference]
Evidence: [Specific quote or data]

Continue for all findings...

IMPORTANT RULES:
- Each finding must cite a specific source using [S#] for summaries or [D#] for detailed sources
- Include specific evidence (quotes, data, page numbers when available)
- Be precise and factual
- Focus on the most significant findings
- Number each finding clearly (Finding 1, Finding 2, etc.)"""

    user_prompt = f"""Analyze these research papers and extract key findings with citations:

========================================
DOCUMENT SUMMARIES:
========================================
{summaries_context}

SUMMARY SOURCES:
{chr(10).join([f"[S{i + 1}] {s.get('source', 'Unknown')}" for i, s in enumerate(summary_sources[:10])])}

========================================
DETAILED CONTEXT:
========================================
{context_text}

DETAILED SOURCES:
{chr(10).join([f"[D{i + 1}] {s.get('source', 'Unknown')} (Page: {s.get('page', 'N/A')})" for i, s in enumerate(sources[:10])])}

Extract 5-7 key findings following the exact format specified in your instructions."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        analysis = response.content

        logger.info(f"Raw analysis preview: {analysis[:200]}...")

        # Parse findings
        findings = []
        lines = analysis.split('\n')
        current_finding = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Finding') and ':' in line:
                if current_finding and 'finding' in current_finding:
                    findings.append(current_finding)
                # Extract finding text after the number and colon
                finding_text = line.split(':', 1)[1].strip() if ':' in line else line
                current_finding = {'finding': finding_text}
            elif line.startswith('Citation:'):
                current_finding['citation'] = line.split(':', 1)[1].strip()
            elif line.startswith('Evidence:'):
                current_finding['evidence'] = line.split(':', 1)[1].strip()

        if current_finding and 'finding' in current_finding:
            findings.append(current_finding)

        logger.info(f"RESEARCHER: Extracted {len(findings)} findings")
        if findings:
            logger.info(f"Sample finding: {findings[0]}")

        # Combine all sources
        all_sources = []
        for s in summary_sources:
            source_copy = s.copy()
            source_copy['type'] = 'summary'
            all_sources.append(source_copy)
        for s in sources:
            source_copy = s.copy()
            source_copy['type'] = 'detailed'
            all_sources.append(source_copy)

        return {
            "messages": [AIMessage(content=f"RESEARCHER completed with {len(findings)} findings")],
            "researcher_analysis": analysis,
            "researcher_findings": findings,
            "researcher_sources": all_sources,
            "researcher_status": "success",
            "current_step": 1,
            "workflow_status": "in_progress"
        }

    except Exception as e:
        logger.error(f"RESEARCHER error: {str(e)}")
        return {
            "messages": [AIMessage(content=f"RESEARCHER failed: {str(e)}")],
            "researcher_status": "error",
            "workflow_status": "error",
            "error_message": str(e)
        }


def reviewer_node(state: ResearchState) -> Dict:
    """
    REVIEWER Agent: Critical evaluation of findings.
    """
    logger.info("REVIEWER: Starting critical review...")

    if state.get("researcher_status") != "success":
        return {
            "reviewer_status": "skipped",
            "error_message": "Researcher agent did not complete successfully"
        }

    researcher_analysis = state["researcher_analysis"]
    findings = state["researcher_findings"]

    system_prompt = """You are a critical reviewer evaluating research findings.

YOUR TASK:
Review the researcher's findings and identify:
1. Strengths (3-5 points): What is well-supported and convincing
2. Weaknesses (3-5 points): What is questionable or needs more evidence

OUTPUT FORMAT (use exactly this structure):
STRENGTHS:
1. [Strength statement with brief explanation]
2. [Strength statement with brief explanation]
...

WEAKNESSES:
1. [Weakness statement with brief explanation]
2. [Weakness statement with brief explanation]
...

Be specific and constructive. Focus on the quality of evidence and reasoning."""

    user_prompt = f"""Review these research findings:

RESEARCHER'S FINDINGS:
{chr(10).join([f"- {f.get('finding', 'N/A')} (Citation: {f.get('citation', 'N/A')})" for f in findings])}

FULL ANALYSIS:
{researcher_analysis}

Identify strengths and weaknesses following the exact format specified."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        critique = response.content

        # Parse strengths and weaknesses
        strengths = []
        weaknesses = []
        lines = critique.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.upper().startswith('STRENGTHS'):
                current_section = 'strengths'
            elif line.upper().startswith('WEAKNESSES'):
                current_section = 'weaknesses'
            elif line and (line[0].isdigit() or line.startswith('-')):
                # Remove leading number/dash and period
                text = line.lstrip('0123456789.-) ').strip()
                if text and current_section == 'strengths':
                    strengths.append(text)
                elif text and current_section == 'weaknesses':
                    weaknesses.append(text)

        logger.info(f"REVIEWER: Found {len(strengths)} strengths, {len(weaknesses)} weaknesses")

        return {
            "messages": [AIMessage(content="REVIEWER completed critique")],
            "reviewer_critique": critique,
            "reviewer_strengths": strengths,
            "reviewer_weaknesses": weaknesses,
            "reviewer_status": "success",
            "current_step": 2
        }

    except Exception as e:
        logger.error(f"REVIEWER error: {str(e)}")
        return {
            "messages": [AIMessage(content=f"REVIEWER failed: {str(e)}")],
            "reviewer_status": "error",
            "workflow_status": "error",
            "error_message": str(e)
        }


def synthesizer_node(state: ResearchState) -> Dict:
    """
    SYNTHESIZER Agent: Creates novel hypotheses.
    """
    logger.info("SYNTHESIZER: Generating novel insights...")

    if state.get("reviewer_status") != "success":
        return {
            "synthesizer_status": "skipped",
            "error_message": "Reviewer agent did not complete successfully"
        }

    findings = state["researcher_findings"]
    critique = state["reviewer_critique"]

    system_prompt = """You are a research synthesizer generating novel testable hypotheses.

YOUR TASK:
Based on the research findings and review, generate 3-5 novel hypotheses.

OUTPUT FORMAT (use exactly this structure):
Hypothesis 1: [Clear if-then statement]
Rationale: [Why this is plausible based on findings]
Test: [How to verify this hypothesis]

Hypothesis 2: [Clear if-then statement]
Rationale: [Why this is plausible]
Test: [How to verify]

Continue for all hypotheses...

REQUIREMENTS:
- Each hypothesis must be testable
- Must be novel (not just restating findings)
- Must be based on evidence from the research
- Use clear if-then structure"""

    user_prompt = f"""Generate novel hypotheses based on these findings:

KEY FINDINGS:
{chr(10).join([f"- {f.get('finding', 'N/A')}" for f in findings])}

REVIEW CRITIQUE:
{critique[:500]}...

Generate 3-5 testable hypotheses following the exact format specified."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        synthesis = response.content

        # Parse hypotheses
        hypotheses = []
        lines = synthesis.split('\n')
        current_hypothesis = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Hypothesis') and ':' in line:
                if current_hypothesis and 'hypothesis' in current_hypothesis:
                    hypotheses.append(current_hypothesis)
                hyp_text = line.split(':', 1)[1].strip() if ':' in line else line
                current_hypothesis = {'hypothesis': hyp_text}
            elif line.startswith('Rationale:'):
                current_hypothesis['rationale'] = line.split(':', 1)[1].strip()
            elif line.startswith('Test:'):
                current_hypothesis['test'] = line.split(':', 1)[1].strip()

        if current_hypothesis and 'hypothesis' in current_hypothesis:
            hypotheses.append(current_hypothesis)

        logger.info(f"SYNTHESIZER: Generated {len(hypotheses)} hypotheses")

        return {
            "messages": [AIMessage(content="SYNTHESIZER completed synthesis")],
            "synthesizer_synthesis": synthesis,
            "synthesizer_hypotheses": hypotheses,
            "synthesizer_status": "success",
            "current_step": 3
        }

    except Exception as e:
        logger.error(f"SYNTHESIZER error: {str(e)}")
        return {
            "messages": [AIMessage(content=f"SYNTHESIZER failed: {str(e)}")],
            "synthesizer_status": "error",
            "workflow_status": "error",
            "error_message": str(e)
        }


def questioner_node(state: ResearchState) -> Dict:
    """
    QUESTIONER Agent: Identifies gaps and generates research questions.
    """
    logger.info("QUESTIONER: Identifying gaps and questions...")

    if state.get("synthesizer_status") != "success":
        return {
            "questioner_status": "skipped",
            "error_message": "Synthesizer agent did not complete successfully"
        }

    findings = state["researcher_findings"]
    hypotheses = state["synthesizer_hypotheses"]

    system_prompt = """You are a research questioner identifying critical gaps and formulating research questions.

YOUR TASK:
Based on the findings and hypotheses, generate 5-7 important research questions.

OUTPUT FORMAT (use exactly this structure):
1. [Research question]?
2. [Research question]?
3. [Research question]?
...

REQUIREMENTS:
- Each question must be specific and answerable
- Questions should address important gaps
- Questions should be feasible to investigate
- Number each question clearly"""

    user_prompt = f"""Generate research questions based on:

KEY FINDINGS:
{chr(10).join([f"- {f.get('finding', 'N/A')}" for f in findings])}

HYPOTHESES:
{chr(10).join([f"- {h.get('hypothesis', 'N/A')}" for h in hypotheses])}

Generate 5-7 important research questions following the exact format specified."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.4
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        gap_analysis = response.content

        # Parse questions
        questions = []
        lines = gap_analysis.split('\n')

        for line in lines:
            line = line.strip()
            if '?' in line and len(line) > 20:
                # Remove leading numbers/dashes
                question = line.lstrip('0123456789.-) ').strip()
                if question:
                    questions.append(question)

        logger.info(f"QUESTIONER: Identified {len(questions)} questions")

        return {
            "messages": [AIMessage(content="QUESTIONER completed gap analysis")],
            "questioner_gap_analysis": gap_analysis,
            "questioner_questions": questions,
            "questioner_status": "success",
            "current_step": 4
        }

    except Exception as e:
        logger.error(f"QUESTIONER error: {str(e)}")
        return {
            "messages": [AIMessage(content=f"QUESTIONER failed: {str(e)}")],
            "questioner_status": "error",
            "workflow_status": "error",
            "error_message": str(e)
        }


def formatter_node(state: ResearchState) -> Dict:
    """
    FORMATTER Agent: Compiles comprehensive report.
    """
    logger.info("FORMATTER: Compiling final report...")

    if state.get("questioner_status") != "success":
        return {
            "formatter_status": "skipped",
            "error_message": "Questioner agent did not complete successfully"
        }

    # Gather all data
    findings = state["researcher_findings"]
    strengths = state["reviewer_strengths"]
    weaknesses = state["reviewer_weaknesses"]
    hypotheses = state["synthesizer_hypotheses"]
    questions = state["questioner_questions"]
    sources = state["researcher_sources"]

    system_prompt = """You are a research report compiler creating a comprehensive, well-structured report.

YOUR TASK:
Compile all the research findings, review, hypotheses, and questions into a clear, professional report.

REPORT STRUCTURE:
1. Executive Summary
2. Key Findings (with citations)
3. Critical Review (strengths and weaknesses)
4. Novel Hypotheses
5. Research Questions
6. Sources

Make the report clear, well-organized, and professionally formatted."""

    user_prompt = f"""Compile a comprehensive research report from:

KEY FINDINGS ({len(findings)} total):
{chr(10).join([f"- {f.get('finding', 'N/A')} [{f.get('citation', 'N/A')}]" for f in findings])}

STRENGTHS ({len(strengths)} total):
{chr(10).join([f"- {s}" for s in strengths])}

WEAKNESSES ({len(weaknesses)} total):
{chr(10).join([f"- {w}" for w in weaknesses])}

NOVEL HYPOTHESES ({len(hypotheses)} total):
{chr(10).join([f"- {h.get('hypothesis', 'N/A')}" for h in hypotheses])}

RESEARCH QUESTIONS ({len(questions)} total):
{chr(10).join([f"- {q}" for q in questions])}

SOURCES ({len(sources)} total):
{chr(10).join([f"[{i+1}] {s.get('source', 'Unknown')} ({s.get('type', 'N/A')})" for i, s in enumerate(sources[:15])])}

Create a well-structured, comprehensive research report."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        report = response.content

        logger.info("FORMATTER: Report compiled successfully")

        return {
            "messages": [AIMessage(content="FORMATTER completed comprehensive report")],
            "final_report": report,
            "formatter_status": "success",
            "workflow_status": "success",
            "current_step": 5
        }

    except Exception as e:
        logger.error(f"FORMATTER error: {str(e)}")
        return {
            "messages": [AIMessage(content=f"FORMATTER failed: {str(e)}")],
            "formatter_status": "error",
            "workflow_status": "error",
            "error_message": str(e)
        }


# --- ROUTER FUNCTIONS ---
def route_after_researcher(state: ResearchState) -> str:
    """Route after researcher based on success status"""
    if state.get("researcher_status") == "success":
        return "reviewer"
    return "error_handler"


def route_after_reviewer(state: ResearchState) -> str:
    """Route after reviewer based on success status"""
    if state.get("reviewer_status") == "success":
        return "synthesizer"
    return "error_handler"


def route_after_synthesizer(state: ResearchState) -> str:
    """Route after synthesizer based on success status"""
    if state.get("synthesizer_status") == "success":
        return "questioner"
    return "error_handler"


def route_after_questioner(state: ResearchState) -> str:
    """Route after questioner based on success status"""
    if state.get("questioner_status") == "success":
        return "formatter"
    return "error_handler"


def error_handler_node(state: ResearchState) -> Dict:
    """Handle errors in the workflow"""
    error_msg = state.get("error_message", "Unknown error occurred")
    logger.error(f"Workflow error: {error_msg}")

    return {
        "messages": [AIMessage(content=f"Workflow failed: {error_msg}")],
        "workflow_status": "error"
    }


# --- Build LangGraph ---
def create_research_graph():
    """Create and compile the research workflow graph"""

    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("questioner", questioner_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("error_handler", error_handler_node)

    # Set workflow edges
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "reviewer": "reviewer",
            "error_handler": "error_handler"
        }
    )
    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "synthesizer": "synthesizer",
            "error_handler": "error_handler"
        }
    )
    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {
            "questioner": "questioner",
            "error_handler": "error_handler"
        }
    )
    graph.add_conditional_edges(
        "questioner",
        route_after_questioner,
        {
            "formatter": "formatter",
            "error_handler": "error_handler"
        }
    )
    graph.add_edge("formatter", END)
    graph.add_edge("error_handler", END)

    # Compile with memory
    return graph.compile(checkpointer=memory)


# --- Main Execution ---
def run_research_analysis(config: Optional[Dict] = None):
    """
    Run the complete research analysis workflow.

    Args:
        config: Configuration dict with thread_id for checkpointing

    Returns:
        Final state with research report
    """
    graph = create_research_graph()

    if config is None:
        config = {"configurable": {"thread_id": "research_thread_1"}}

    initial_state = {
        "messages": [],
        "workflow_status": "started",
        "current_step": 0,
        "researcher_status": "",
        "reviewer_status": "",
        "synthesizer_status": "",
        "questioner_status": "",
        "formatter_status": "",
        "error_message": ""
    }

    print(f"\n{'=' * 70}")
    print("MULTI-AGENT RESEARCH SYSTEM")
    print(f"{'=' * 70}\n")

    # Run the graph
    result = graph.invoke(initial_state, config)

    if result["workflow_status"] == "success":
        print(f"\n{'=' * 70}")
        print("WORKFLOW COMPLETE")
        print(f"{'=' * 70}")

        print(f"\n{'=' * 70}")
        print("FINAL REPORT")
        print(f"{'=' * 70}\n")
        print(result["final_report"])
        print(f"\n{'=' * 70}\n")

    else:
        print(f"\n❌ WORKFLOW FAILED: {result.get('error_message', 'Unknown error')}\n")

    return result


if __name__ == "__main__":
    # Run analysis
    result = run_research_analysis()

    # Save report to file
    if result["workflow_status"] == "success":
        with open("research_report.txt", "w") as f:
            f.write("="*70 + "\n")
            f.write("MULTI-AGENT RESEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(result["final_report"])

        print("✓ Report saved to research_report.txt")