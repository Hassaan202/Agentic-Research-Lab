"""
LangGraph-based Multi-Agent Research System
Converts the sequential agent pipeline into a graph-based workflow
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
from langchain_core.tools import tool

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



# --- STATE SCHEMA ---
class ResearchState(TypedDict):
    """State for the multi-agent research workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Researcher outputs
    researcher_analysis: str
    researcher_findings: List[str]
    researcher_sources: List[Dict]
    researcher_status: str

    # Reviewer outputs
    reviewer_critique: str
    reviewer_strengths: List[str]
    reviewer_weaknesses: List[str]
    reviewer_status: str

    # Synthesizer outputs
    synthesizer_synthesis: str
    synthesizer_hypotheses: List[str]
    synthesizer_insights: List[str]
    synthesizer_status: str

    # Questioner outputs
    questioner_gap_analysis: str
    questioner_gaps: List[str]
    questioner_questions: List[str]
    questioner_status: str

    # Formatter outputs
    final_report: str
    formatter_status: str

    # Workflow metadata
    workflow_status: str
    error_message: str
    current_step: int




# --- RAG TOOL ---
@tool
def retrieve_context_tool(query: str, k: int = 8) -> str:
    """
    Retrieve relevant context from the RAG pipeline.

    Args:
        query: Search query for document retrieval
        k: Number of documents to retrieve

    Returns:
        Retrieved context as string
    """
    result = rag_pipeline.answer_question(query, k=k, return_sources=True)
    return result['answer']




# --- AGENT NODES ---
def researcher_node(state: ResearchState) -> Dict:
    """
    RESEARCHER Agent: Analyzes research papers and extracts key findings.
    """
    logger.info("RESEARCHER: Starting analysis of research papers...")

    # Retrieve context from RAG
    context_result = rag_pipeline.answer_question(
        "provide a comprehensive analysis of the documents",
        k=10,
        return_sources=True
    )
    context_text = context_result['answer']
    sources = context_result.get('sources', [])

    system_prompt = """You are a meticulous research analyst. Your task is to analyze research papers and extract key findings, methodologies, and conclusions.

CRITICAL RULES:
1. ONLY use information from the provided context - DO NOT hallucinate or invent facts
2. Cite specific sources for every finding you mention
3. Extract key methodologies, results, and conclusions
4. Identify the main contributions of each paper
5. Note any limitations or gaps mentioned in the papers
6. Be precise and factual - avoid speculation

Format your analysis clearly with sections for:
- Key Findings
- Methodologies
- Conclusions
- Limitations
"""

    user_prompt = f"""Analyze the following research papers:

CONTEXT FROM DOCUMENTS:
{context_text}

SOURCES:
{chr(10).join([f"- {s.get('source', 'Unknown')} (Page: {s.get('page', 'N/A')})" for s in sources[:5]])}

Please provide a detailed analysis with:
1. Key findings from the papers
2. Methodologies used
3. Main conclusions
4. Any limitations or gaps mentioned

Remember: Only use information from the provided context. Cite sources for each finding."""

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
        analysis = response.content

        # Extract findings (simplified)
        findings = [line.strip() for line in analysis.split('\n')
                    if line.strip() and (line.strip().startswith('-') or
                                         (line.strip() and line.strip()[0].isdigit()))][:10]

        logger.info(f"RESEARCHER: Analysis complete with {len(findings)} findings")

        return {
            "messages": [AIMessage(content=f"RESEARCHER completed analysis")],
            "researcher_analysis": analysis,
            "researcher_findings": findings,
            "researcher_sources": sources,
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
    REVIEWER Agent: Critiques findings and identifies strengths/weaknesses.
    """
    logger.info("REVIEWER: Starting critique of findings...")

    if state.get("researcher_status") != "success":
        return {
            "reviewer_status": "skipped",
            "error_message": "Researcher agent did not complete successfully"
        }

    researcher_analysis = state["researcher_analysis"]
    findings = state["researcher_findings"]

    # Retrieve additional context for critique
    context_result = rag_pipeline.answer_question(
        "methodology limitations weaknesses",
        k=6,
        return_sources=True
    )
    additional_context = context_result.get('answer', 'No additional context available')

    system_prompt = """You are a critical research reviewer. Your task is to evaluate research findings, identify strengths, weaknesses, and potential biases.

CRITICAL RULES:
1. Base your critique ONLY on the provided context and findings
2. Identify methodological strengths and weaknesses
3. Look for potential biases or limitations
4. Check for consistency and logical coherence
5. Identify any gaps in the analysis
6. Be constructive and specific - cite sources when possible
7. DO NOT make up criticisms that aren't supported by the context

Format your critique with:
- Strengths
- Weaknesses
- Potential Biases
- Gaps or Missing Information
"""

    user_prompt = f"""Review and critique the following research analysis:

RESEARCHER'S ANALYSIS:
{researcher_analysis}

KEY FINDINGS:
{chr(10).join([f"- {f}" for f in findings[:10]])}

ADDITIONAL CONTEXT:
{additional_context}

Please provide a thorough critique focusing on:
1. Strengths of the research and analysis
2. Weaknesses or limitations
3. Potential biases or methodological concerns
4. Gaps in the analysis or missing information
5. Consistency and logical coherence

Remember: Base your critique on the actual content. Do not invent criticisms."""

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

        # Extract strengths and weaknesses (simplified)
        lines = critique.split('\n')
        strengths = []
        weaknesses = []

        in_strengths = False
        in_weaknesses = False

        for line in lines:
            if 'strength' in line.lower():
                in_strengths = True
                in_weaknesses = False
            elif 'weakness' in line.lower():
                in_weaknesses = True
                in_strengths = False
            elif line.strip() and line.strip().startswith('-'):
                if in_strengths:
                    strengths.append(line.strip())
                elif in_weaknesses:
                    weaknesses.append(line.strip())

        logger.info("REVIEWER: Critique complete")

        return {
            "messages": [AIMessage(content="REVIEWER completed critique")],
            "reviewer_critique": critique,
            "reviewer_strengths": strengths[:5],
            "reviewer_weaknesses": weaknesses[:5],
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
    SYNTHESIZER Agent: Synthesizes insights and generates hypotheses.
    """
    logger.info("SYNTHESIZER: Starting synthesis...")

    if state.get("reviewer_status") != "success":
        return {
            "synthesizer_status": "skipped",
            "error_message": "Reviewer agent did not complete successfully"
        }

    researcher_analysis = state["researcher_analysis"]
    critique = state["reviewer_critique"]
    strengths = state["reviewer_strengths"]
    weaknesses = state["reviewer_weaknesses"]

    # Retrieve context for synthesis
    context_result = rag_pipeline.answer_question(
        "hypotheses research questions future work",
        k=6,
        return_sources=True
    )
    additional_context = context_result.get('answer', 'No additional context available')

    system_prompt = """You are a research synthesizer. Your task is to combine findings and critiques to generate new insights and testable hypotheses.

CRITICAL RULES:
1. Base hypotheses ONLY on the provided findings and context
2. Generate testable, specific hypotheses
3. Connect findings from different sources
4. Identify patterns and relationships
5. Propose actionable research directions
6. DO NOT create hypotheses that aren't supported by the evidence
7. Clearly state what evidence supports each hypothesis

Format your synthesis with:
- Key Insights
- Patterns and Relationships
- Testable Hypotheses
- Research Directions
"""

    user_prompt = f"""Synthesize the following research analysis and critique:

RESEARCHER'S FINDINGS:
{researcher_analysis}

REVIEWER'S CRITIQUE:
{critique}

STRENGTHS IDENTIFIED:
{chr(10).join([f"- {s}" for s in strengths[:5]])}

WEAKNESSES IDENTIFIED:
{chr(10).join([f"- {w}" for w in weaknesses[:5]])}

ADDITIONAL CONTEXT:
{additional_context}

Please synthesize this information to:
1. Identify key insights and patterns
2. Connect findings from different sources
3. Generate 3-5 testable hypotheses
4. Propose specific research directions
5. Explain the evidence base for each hypothesis

Remember: Hypotheses must be grounded in the actual findings. Be specific and testable."""

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
        synthesis = response.content

        # Extract hypotheses and insights
        hypotheses = []
        insights = []
        lines = synthesis.split('\n')

        for line in lines:
            if 'hypothesis' in line.lower() or 'h1' in line.lower() or 'h2' in line.lower():
                if line.strip() and len(line.strip()) > 20:
                    hypotheses.append(line.strip())
            elif ('insight' in line.lower() or 'pattern' in line.lower()) and line.strip():
                if len(line.strip()) > 30:
                    insights.append(line.strip())

        logger.info(f"SYNTHESIZER: Generated {len(hypotheses)} hypotheses")

        return {
            "messages": [AIMessage(content="SYNTHESIZER completed synthesis")],
            "synthesizer_synthesis": synthesis,
            "synthesizer_hypotheses": hypotheses[:5],
            "synthesizer_insights": insights[:5],
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
    QUESTIONER Agent: Identifies gaps and generates follow-up questions.
    """
    logger.info("QUESTIONER: Identifying gaps and generating questions...")

    if state.get("synthesizer_status") != "success":
        return {
            "questioner_status": "skipped",
            "error_message": "Synthesizer agent did not complete successfully"
        }

    synthesis = state["synthesizer_synthesis"]
    hypotheses = state["synthesizer_hypotheses"]
    insights = state["synthesizer_insights"]
    researcher_analysis = state["researcher_analysis"]
    critique = state["reviewer_critique"]

    # Retrieve context for gap identification
    context_result = rag_pipeline.answer_question(
        "research gaps limitations future work",
        k=5,
        return_sources=True
    )
    additional_context = context_result.get('answer', 'No additional context available')

    system_prompt = """You are a research questioner. Your task is to identify knowledge gaps and generate critical follow-up questions.

CRITICAL RULES:
1. Identify gaps based on the actual analysis and synthesis provided
2. Generate specific, answerable research questions
3. Focus on gaps that are evident from the research
4. Prioritize questions that would advance the field
5. Ensure questions are grounded in the existing research
6. DO NOT create questions about topics not related to the research

Format your output with:
- Knowledge Gaps
- Critical Questions
- Research Priorities
"""

    user_prompt = f"""Identify gaps and generate questions based on the following research analysis:

SYNTHESIS AND HYPOTHESES:
{synthesis}

HYPOTHESES GENERATED:
{chr(10).join([f"- {h}" for h in hypotheses[:5]])}

KEY INSIGHTS:
{chr(10).join([f"- {i}" for i in insights[:5]])}

RESEARCHER'S ANALYSIS:
{researcher_analysis[:500]}...

REVIEWER'S CRITIQUE:
{critique[:500]}...

ADDITIONAL CONTEXT:
{additional_context}

Please identify:
1. Knowledge gaps in the current research
2. Unanswered questions that emerged
3. Critical follow-up questions (5-7 questions)
4. Research priorities for future work
5. Areas needing further investigation

Remember: Questions should be specific and answerable. Base gaps on actual limitations identified."""

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

        # Extract gaps and questions
        gaps = []
        questions = []
        lines = gap_analysis.split('\n')

        for line in lines:
            if 'gap' in line.lower() and line.strip() and len(line.strip()) > 20:
                gaps.append(line.strip())
            elif '?' in line and line.strip() and len(line.strip()) > 10:
                questions.append(line.strip())

        logger.info(f"QUESTIONER: Identified {len(questions)} research questions")

        return {
            "messages": [AIMessage(content="QUESTIONER completed gap analysis")],
            "questioner_gap_analysis": gap_analysis,
            "questioner_gaps": gaps[:5],
            "questioner_questions": questions[:7],
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
    FORMATTER Agent: Compiles the final research report.
    """
    logger.info("FORMATTER: Compiling final report...")

    if state.get("questioner_status") != "success":
        return {
            "formatter_status": "skipped",
            "error_message": "Questioner agent did not complete successfully"
        }

    researcher_analysis = state["researcher_analysis"]
    critique = state["reviewer_critique"]
    synthesis = state["synthesizer_synthesis"]
    hypotheses = state["synthesizer_hypotheses"]
    gap_analysis = state["questioner_gap_analysis"]
    questions = state["questioner_questions"]
    sources = state["researcher_sources"]

    system_prompt = """You are a research report formatter. Your task is to compile a comprehensive, well-structured research report from multiple agent analyses.

CRITICAL RULES:
1. Organize information clearly and logically
2. Include all key findings, critiques, hypotheses, and questions
3. Maintain accuracy - only include information from the provided analyses
4. Use proper citations and source references
5. Create a professional, readable format
6. Include executive summary and detailed sections
7. DO NOT add information that wasn't in the original analyses

Format the report with:
- Executive Summary
- Key Findings
- Critical Analysis
- Synthesized Insights
- Hypotheses
- Research Gaps and Questions
- Conclusions
- Sources
"""

    user_prompt = f"""Compile a comprehensive research report:

RESEARCHER'S ANALYSIS:
{researcher_analysis}

REVIEWER'S CRITIQUE:
{critique}

SYNTHESIZER'S SYNTHESIS:
{synthesis}

HYPOTHESES:
{chr(10).join([f"- {h}" for h in hypotheses])}

QUESTIONER's GAP ANALYSIS:
{gap_analysis}

RESEARCH QUESTIONS:
{chr(10).join([f"- {q}" for q in questions])}

SOURCES:
{chr(10).join([f"- {s.get('source', 'Unknown')}" for s in sources[:10]])}

Please compile a comprehensive research report with proper structure, citations, and all key information from the analyses above."""

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
        report = response.content

        logger.info("FORMATTER: Report compiled successfully")

        return {
            "messages": [AIMessage(content="FORMATTER completed report")],
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
    print("MULTI-AGENT RESEARCH LAB")
    print(f"{'=' * 70}\n")

    # Run the graph
    result = graph.invoke(initial_state, config)

    if result["workflow_status"] == "success":
        print(f"\n{'=' * 70}")
        print("WORKFLOW COMPLETE")
        print(f"{'=' * 70}")
        print(f"\nFINAL REPORT:\n")
        print(result["final_report"])
        print(f"\n{'=' * 70}\n")
    else:
        print(f"\n❌ WORKFLOW FAILED: {result.get('error_message', 'Unknown error')}\n")

    return result


if __name__ == "__main__":
    # Example usage
    result = run_research_analysis()

    # Save report to file
    if result["workflow_status"] == "success":
        with open("langgraph_research_report.txt", "w") as f:
            f.write(result["final_report"])
        print("✓ Report saved to langgraph_research_report.txt")