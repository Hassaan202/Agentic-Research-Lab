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


# --- ENHANCED STATE SCHEMA ---
class ResearchState(TypedDict):
    """Enhanced state with reasoning traces and verification"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Researcher outputs
    researcher_analysis: str
    researcher_findings: List[Dict]  # Changed to Dict for citations
    researcher_sources: List[Dict]
    researcher_status: str
    researcher_reasoning: str  # NEW: Reasoning trace
    researcher_confidence: float  # NEW: Confidence score

    # Reviewer outputs
    reviewer_critique: str
    reviewer_strengths: List[Dict]  # Changed to Dict for evidence
    reviewer_weaknesses: List[Dict]  # Changed to Dict for evidence
    reviewer_status: str
    reviewer_reasoning: str  # NEW: Reasoning trace
    reviewer_questions_to_researcher: List[str]  # NEW: Inter-agent questions

    # Synthesizer outputs
    synthesizer_synthesis: str
    synthesizer_hypotheses: List[Dict]  # Changed to Dict for evidence
    synthesizer_insights: List[Dict]  # Changed to Dict for evidence
    synthesizer_status: str
    synthesizer_reasoning: str  # NEW: Reasoning trace
    synthesizer_novel_connections: List[Dict]  # NEW: Novel insights

    # Questioner outputs
    questioner_gap_analysis: str
    questioner_gaps: List[Dict]  # Changed to Dict for context
    questioner_questions: List[Dict]  # Changed to Dict for reasoning
    questioner_status: str
    questioner_reasoning: str  # NEW: Reasoning trace

    # Formatter outputs
    final_report: str
    formatter_status: str
    reasoning_graph: Dict  # NEW: Agent conversation flow

    # Workflow metadata
    workflow_status: str
    error_message: str
    current_step: int
    agent_interactions: List[Dict]  # NEW: Track agent conversations
    verification_scores: Dict  # NEW: Track verification metrics




# ---AGENT NODES ---
def researcher_node(state: ResearchState) -> Dict:
    """
    RESEARCHER Agent: Analyzes research papers with explicit reasoning.
    """
    logger.info("RESEARCHER: Starting analysis with reasoning traces...")

    # Retrieve context from RAG with more documents for better coverage
    context_result = rag_pipeline.answer_question(
        "provide a comprehensive analysis of the documents including key findings, methodologies, and conclusions",
        k=10,  # Increased for better coverage
        return_sources=True
    )
    context_text = context_result['answer']
    sources = context_result.get('sources', [])

    system_prompt = """You are a meticulous research analyst with expertise in critical thinking and evidence-based reasoning.

 CRITICAL RULES FOR REASONING:
1. ALWAYS cite specific sources with page numbers for EVERY claim
2. Provide explicit reasoning chains: "Because X (source), we can infer Y"
3. Rate your confidence (0-100%) for each major finding
4. Distinguish between: established facts, probable inferences, and speculative connections
5. Flag any contradictions or uncertainties in the literature
6. Note when evidence is insufficient or when claims need verification

STRUCTURED OUTPUT FORMAT:
For each finding, provide:
- Finding: [Clear statement]
- Evidence: [Specific citation with page/section]
- Reasoning: [Why this evidence supports the finding]
- Confidence: [0-100%]
- Limitations: [What could undermine this finding]

Your analysis should be:
- Methodical: Build arguments step-by-step
- Transparent: Show your reasoning process
- Verifiable: Every claim traceable to sources
- Honest: Acknowledge gaps and uncertainties
"""

    user_prompt = f"""Analyze the research papers with explicit reasoning:

CONTEXT FROM DOCUMENTS:
{context_text}

AVAILABLE SOURCES:
{chr(10).join([f"[{i+1}] {s.get('source', 'Unknown')} (Page: {s.get('page', 'N/A')})" for i, s in enumerate(sources[:10])])}

ANALYSIS REQUIREMENTS:
1. Extract 5-7 key findings with:
   - Specific evidence (cite source number)
   - Reasoning chain
   - Confidence score
   
2. Identify methodologies used:
   - What approaches were taken
   - Why they were chosen
   - Limitations of each method
   
3. Draw conclusions:
   - What can we reliably conclude?
   - What remains uncertain?
   - What contradictions exist?

4. Provide your overall reasoning process:
   - How you evaluated the evidence
   - Why you prioritized certain findings
   - What assumptions you made

Remember: Show your work. Every claim needs a source and reasoning."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1  # Lower for more factual accuracy
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        analysis = response.content

        # Extract structured findings with evidence
        findings = []
        confidence_scores = []
        reasoning_trace = []

        lines = analysis.split('\n')
        current_finding = {}

        for line in lines:
            line = line.strip()
            if line.startswith('- Finding:') or line.startswith('Finding:'):
                if current_finding:
                    findings.append(current_finding)
                current_finding = {'finding': line.split(':', 1)[1].strip()}
            elif line.startswith('- Evidence:') or line.startswith('Evidence:'):
                current_finding['evidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('- Reasoning:') or line.startswith('Reasoning:'):
                current_finding['reasoning'] = line.split(':', 1)[1].strip()
                reasoning_trace.append(current_finding['reasoning'])
            elif line.startswith('- Confidence:') or line.startswith('Confidence:'):
                conf_str = line.split(':', 1)[1].strip().rstrip('%')
                try:
                    conf = float(conf_str)
                    current_finding['confidence'] = conf
                    confidence_scores.append(conf)
                except:
                    current_finding['confidence'] = 70.0
            elif line.startswith('- Limitations:') or line.startswith('Limitations:'):
                current_finding['limitations'] = line.split(':', 1)[1].strip()

        if current_finding:
            findings.append(current_finding)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 70.0

        logger.info(f"RESEARCHER: Analysis complete with {len(findings)} findings (avg confidence: {avg_confidence:.1f}%)")

        # Create interaction record
        interaction = {
            "from": "RESEARCHER",
            "to": "ALL",
            "type": "analysis",
            "content": f"Completed analysis with {len(findings)} findings",
            "timestamp": "step_1"
        }

        return {
            "messages": [AIMessage(content=f"RESEARCHER completed analysis with {len(findings)} findings")],
            "researcher_analysis": analysis,
            "researcher_findings": findings[:10],
            "researcher_sources": sources,
            "researcher_status": "success",
            "researcher_reasoning": "\n".join(reasoning_trace),
            "researcher_confidence": avg_confidence,
            "current_step": 1,
            "workflow_status": "in_progress",
            "agent_interactions": [interaction]
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
    REVIEWER Agent: Critical evaluation with counter-arguments.
    """
    logger.info("REVIEWER: Starting critical review...")

    if state.get("researcher_status") != "success":
        return {
            "reviewer_status": "skipped",
            "error_message": "Researcher agent did not complete successfully"
        }

    researcher_analysis = state["researcher_analysis"]
    findings = state["researcher_findings"]
    researcher_confidence = state.get("researcher_confidence", 70.0)

    # Retrieve additional context specifically for critique
    context_result = rag_pipeline.answer_question(
        "methodology limitations weaknesses contradictions alternative interpretations",
        k=8,
        return_sources=True
    )
    additional_context = context_result.get('answer', 'No additional context available')

    system_prompt = """You are a critical research reviewer specializing in identifying flaws, biases, and alternative interpretations.

YOUR MISSION:
1. Challenge every major claim - what's the counter-evidence?
2. Identify methodological weaknesses
3. Question assumptions and logical leaps
4. Propose alternative interpretations
5. Generate tough questions for the researcher

CRITICAL THINKING FRAMEWORK:
- For each finding: "What could disprove this?"
- For methods: "What biases could this introduce?"
- For conclusions: "What alternative explanations exist?"
- For evidence: "Is this cherry-picked? What's missing?"

OUTPUT STRUCTURE:
Strengths: [What's genuinely solid, with evidence]
Weaknesses: [Specific flaws, with reasoning]
Alternative Interpretations: [Other ways to read the evidence]
Questions for Researcher: [Tough questions to probe gaps]
Confidence Assessment: [Is the researcher over/under confident?]

Be constructive but rigorous. Your job is to strengthen the analysis through critique."""

    user_prompt = f"""Critically review this research analysis:

RESEARCHER'S ANALYSIS:
{researcher_analysis}

KEY FINDINGS:
{chr(10).join([f"- {f.get('finding', 'N/A')} (Confidence: {f.get('confidence', 'N/A')}%)" for f in findings[:10]])}

RESEARCHER'S CONFIDENCE: {researcher_confidence:.1f}%

ADDITIONAL CONTEXT FOR CRITIQUE:
{additional_context}

YOUR TASKS:
1. Identify 3-5 strengths (what's well-supported)
2. Identify 3-5 weaknesses (what's questionable)
3. Propose 2-3 alternative interpretations
4. Generate 3-5 tough questions for the researcher
5. Assess if confidence levels are justified

For each point, provide:
- The claim being evaluated
- Your reasoning
- Supporting evidence or counter-evidence
- Confidence in your critique"""

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

        # Extract structured strengths and weaknesses
        lines = critique.split('\n')
        strengths = []
        weaknesses = []
        questions = []
        reasoning_trace = []

        current_section = None
        current_item = {}

        for line in lines:
            line = line.strip()

            if 'strength' in line.lower() and ':' in line:
                current_section = 'strength'
            elif 'weakness' in line.lower() and ':' in line:
                current_section = 'weakness'
            elif '?' in line and len(line) > 20:
                questions.append(line)
            elif line.startswith('-') and current_section:
                if current_item and 'claim' in current_item:
                    if current_section == 'strength':
                        strengths.append(current_item)
                    elif current_section == 'weakness':
                        weaknesses.append(current_item)
                    reasoning_trace.append(current_item.get('reasoning', ''))

                current_item = {'claim': line[1:].strip()}
            elif line.lower().startswith('reasoning:') and current_item:
                current_item['reasoning'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('evidence:') and current_item:
                current_item['evidence'] = line.split(':', 1)[1].strip()

        if current_item and 'claim' in current_item:
            if current_section == 'strength':
                strengths.append(current_item)
            elif current_section == 'weakness':
                weaknesses.append(current_item)

        logger.info(f"REVIEWER: Found {len(strengths)} strengths, {len(weaknesses)} weaknesses, {len(questions)} questions")

        # Create interaction record
        interaction = {
            "from": "REVIEWER",
            "to": "RESEARCHER",
            "type": "critique",
            "content": f"Identified {len(weaknesses)} weaknesses and {len(questions)} questions",
            "questions": questions[:5],
            "timestamp": "step_2"
        }

        return {
            "messages": [AIMessage(content="REVIEWER completed critique")],
            "reviewer_critique": critique,
            "reviewer_strengths": strengths[:5],
            "reviewer_weaknesses": weaknesses[:5],
            "reviewer_status": "success",
            "reviewer_reasoning": "\n".join(reasoning_trace),
            "reviewer_questions_to_researcher": questions[:5],
            "current_step": 2,
            "agent_interactions": state.get("agent_interactions", []) + [interaction]
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
    SYNTHESIZER Agent: Creates novel connections and testable hypotheses.
    """
    logger.info("SYNTHESIZER: Generating novel insights...")

    if state.get("reviewer_status") != "success":
        return {
            "synthesizer_status": "skipped",
            "error_message": "Reviewer agent did not complete successfully"
        }

    researcher_analysis = state["researcher_analysis"]
    findings = state["researcher_findings"]
    critique = state["reviewer_critique"]
    strengths = state["reviewer_strengths"]
    weaknesses = state["reviewer_weaknesses"]

    # Retrieve context for synthesis and hypothesis generation
    context_result = rag_pipeline.answer_question(
        "hypotheses research questions future directions novel applications cross-domain insights",
        k=8,
        return_sources=True
    )
    additional_context = context_result.get('answer', 'No additional context available')

    system_prompt = """You are a research synthesizer who identifies novel patterns and generates breakthrough hypotheses.

YOUR UNIQUE ROLE:
1. Find NON-OBVIOUS connections between findings
2. Generate TESTABLE hypotheses (not vague ideas)
3. Bridge different domains or methods
4. Identify emergent patterns that weren't explicit
5. Propose concrete next steps

SYNTHESIS FRAMEWORK:
- Cross-pollination: Connect ideas that weren't linked before
- Elevation: What higher-level principle emerges?
- Inversion: What if we flip the assumption?
- Application: How could this extend to new domains?

HYPOTHESIS REQUIREMENTS:
Each hypothesis must have:
- Clear statement: "If X, then Y"
- Supporting evidence: Why this is plausible
- Testability: How to verify/falsify
- Novelty: Why this isn't obvious
- Impact: Why this matters

REASONING TRANSPARENCY:
Show your synthesis process:
1. What patterns did you notice?
2. Why are they significant?
3. What assumptions are you making?
4. What would need to be true for your hypothesis to hold?"""

    user_prompt = f"""Synthesize findings into novel insights:

RESEARCHER'S FINDINGS:
{chr(10).join([f"- {f.get('finding', 'N/A')} [Confidence: {f.get('confidence', 'N/A')}%]" for f in findings[:10]])}

REVIEWER'S CRITIQUE:
Strengths: {chr(10).join([s.get('claim', 'N/A') for s in strengths[:3]])}
Weaknesses: {chr(10).join([w.get('claim', 'N/A') for w in weaknesses[:3]])}

ADDITIONAL CONTEXT:
{additional_context}

YOUR TASKS:
1. Identify 3-4 NOVEL insights (not just summaries)
   - What patterns emerge across findings?
   - What connections weren't explicit?
   
2. Generate 3-5 TESTABLE hypotheses:
   - Clear if-then statements
   - Evidence base
   - How to test
   - Why it's novel
   
3. Propose concrete research directions:
   - What experiments to run?
   - What data to collect?
   - What collaborations to pursue?

4. Show your reasoning:
   - How did you connect the dots?
   - What assumptions are you making?
   - What could invalidate your synthesis?

Focus on NOVELTY and TESTABILITY, not just summarization."""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5  # Higher for creative synthesis
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        synthesis = response.content

        # Extract structured hypotheses and insights
        hypotheses = []
        insights = []
        novel_connections = []
        reasoning_trace = []
        lines = synthesis.split('\n')

        current_item = {}
        current_type = None

        for line in lines:
            line = line.strip()

            if 'hypothesis' in line.lower() and (':' in line or line.endswith('hypothesis')):
                if current_item and current_type == 'hypothesis':
                    hypotheses.append(current_item)
                    reasoning_trace.append(current_item.get('reasoning', ''))
                current_item = {}
                current_type = 'hypothesis'
            elif 'insight' in line.lower() and (':' in line or line.endswith('insight')):
                if current_item and current_type == 'insight':
                    insights.append(current_item)
                current_item = {}
                current_type = 'insight'
            elif line.startswith('-') and current_type:
                if 'statement' not in current_item:
                    current_item['statement'] = line[1:].strip()
            elif line.lower().startswith('evidence:') and current_item:
                current_item['evidence'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('testability:') and current_item:
                current_item['testability'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('novelty:') and current_item:
                current_item['novelty'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('reasoning:') and current_item:
                current_item['reasoning'] = line.split(':', 1)[1].strip()

        if current_item and current_type:
            if current_type == 'hypothesis':
                hypotheses.append(current_item)
            elif current_type == 'insight':
                insights.append(current_item)

        # Extract novel connections
        for i, h in enumerate(hypotheses):
            if h.get('novelty'):
                novel_connections.append({
                    'connection': h.get('statement', ''),
                    'why_novel': h.get('novelty', ''),
                    'evidence': h.get('evidence', '')
                })

        logger.info(f"SYNTHESIZER: Generated {len(hypotheses)} hypotheses, {len(insights)} insights")

        # Create interaction record
        interaction = {
            "from": "SYNTHESIZER",
            "to": "ALL",
            "type": "synthesis",
            "content": f"Generated {len(hypotheses)} novel hypotheses",
            "novel_contributions": len(novel_connections),
            "timestamp": "step_3"
        }

        return {
            "messages": [AIMessage(content="SYNTHESIZER completed synthesis")],
            "synthesizer_synthesis": synthesis,
            "synthesizer_hypotheses": hypotheses[:5],
            "synthesizer_insights": insights[:5],
            "synthesizer_status": "success",
            "synthesizer_reasoning": "\n".join(reasoning_trace),
            "synthesizer_novel_connections": novel_connections[:5],
            "current_step": 3,
            "agent_interactions": state.get("agent_interactions", []) + [interaction]
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
    QUESTIONER Agent: Identifies critical gaps and generates research questions.
    """
    logger.info("QUESTIONER: Identifying gaps and questions...")

    if state.get("synthesizer_status") != "success":
        return {
            "questioner_status": "skipped",
            "error_message": "Synthesizer agent did not complete successfully"
        }

    synthesis = state["synthesizer_synthesis"]
    hypotheses = state["synthesizer_hypotheses"]
    insights = state["synthesizer_insights"]
    reviewer_questions = state.get("reviewer_questions_to_researcher", [])

    # Retrieve context for gap identification
    context_result = rag_pipeline.answer_question(
        "research gaps open problems limitations unresolved questions future work",
        k=6,
        return_sources=True
    )
    additional_context = context_result.get('answer', 'No additional context available')

    system_prompt = """You are a research questioner who identifies critical gaps and formulates powerful research questions.

YOUR EXPERTISE:
1. Spot what's MISSING (not just what's there)
2. Identify contradictions that need resolution
3. Find assumptions that need testing
4. Generate questions that advance the field

QUESTION QUALITY CRITERIA:
- Specific: Clear what needs to be investigated
- Answerable: Feasible to address empirically
- Important: Would matter if answered
- Novel: Not already well-studied
- Clear: No ambiguity in what's being asked

GAP ANALYSIS FRAMEWORK:
- Empirical gaps: What data is missing?
- Methodological gaps: What tools don't exist?
- Theoretical gaps: What explanations are lacking?
- Application gaps: Where hasn't this been tried?

For each question, provide:
- The question itself
- Why it matters
- What gap it addresses
- How it could be approached
- What impact answering it would have"""

    user_prompt = f"""Identify critical gaps and generate research questions:

SYNTHESIZED HYPOTHESES:
{chr(10).join([f"- {h.get('statement', 'N/A')}" for h in hypotheses[:5]])}

KEY INSIGHTS:
{chr(10).join([f"- {i.get('statement', 'N/A')}" for i in insights[:5]])}

REVIEWER'S UNANSWERED QUESTIONS:
{chr(10).join([f"- {q}" for q in reviewer_questions[:5]])}

ADDITIONAL CONTEXT:
{additional_context}

YOUR TASKS:
1. Identify 3-5 critical knowledge gaps:
   - What's missing from current research?
   - What contradictions need resolution?
   - What assumptions need testing?

2. Generate 5-7 powerful research questions:
   - Each must be specific and answerable
   - Each must address an important gap
   - Prioritize questions by potential impact

3. For top 3 questions, provide:
   - Why this question matters
   - What gap it addresses
   - Suggested approach to answer it
   - Expected impact if answered

4. Show your reasoning:
   - How did you identify these gaps?
   - Why these questions over others?
   - What assumptions are you making about importance?

Focus on questions that would genuinely advance understanding."""

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

        # Extract structured gaps and questions
        gaps = []
        questions = []
        reasoning_trace = []
        lines = gap_analysis.split('\n')

        current_item = {}
        current_type = None

        for line in lines:
            line = line.strip()

            if 'gap' in line.lower() and ':' in line and len(line) > 20:
                if current_item and current_type == 'gap':
                    gaps.append(current_item)
                current_item = {'gap': line.split(':', 1)[1].strip() if ':' in line else line}
                current_type = 'gap'
            elif '?' in line and len(line) > 15:
                if current_item and current_type == 'question':
                    questions.append(current_item)
                    reasoning_trace.append(current_item.get('reasoning', ''))
                current_item = {'question': line}
                current_type = 'question'
            elif line.lower().startswith('why:') or line.lower().startswith('importance:'):
                if current_item:
                    current_item['importance'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('gap addressed:'):
                if current_item:
                    current_item['gap_addressed'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('approach:'):
                if current_item:
                    current_item['approach'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('reasoning:'):
                if current_item:
                    current_item['reasoning'] = line.split(':', 1)[1].strip()

        if current_item:
            if current_type == 'gap':
                gaps.append(current_item)
            elif current_type == 'question':
                questions.append(current_item)

        logger.info(f"QUESTIONER: Identified {len(gaps)} gaps, {len(questions)} questions")

        # Create interaction record
        interaction = {
            "from": "QUESTIONER",
            "to": "ALL",
            "type": "gap_analysis",
            "content": f"Identified {len(gaps)} gaps and {len(questions)} research questions",
            "priority_questions": [q.get('question', '') for q in questions[:3]],
            "timestamp": "step_4"
        }

        return {
            "messages": [AIMessage(content="QUESTIONER completed gap analysis")],
            "questioner_gap_analysis": gap_analysis,
            "questioner_gaps": gaps[:5],
            "questioner_questions": questions[:7],
            "questioner_status": "success",
            "questioner_reasoning": "\n".join(reasoning_trace),
            "current_step": 4,
            "agent_interactions": state.get("agent_interactions", []) + [interaction]
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
    FORMATTER Agent: Compiles comprehensive report with reasoning traces.
    """
    logger.info("FORMATTER: Compiling final report with reasoning...")

    if state.get("questioner_status") != "success":
        return {
            "formatter_status": "skipped",
            "error_message": "Questioner agent did not complete successfully"
        }

    # Gather all data
    findings = state["researcher_findings"]
    researcher_reasoning = state.get("researcher_reasoning", "")
    researcher_confidence = state.get("researcher_confidence", 0)

    strengths = state["reviewer_strengths"]
    weaknesses = state["reviewer_weaknesses"]
    reviewer_reasoning = state.get("reviewer_reasoning", "")
    reviewer_questions = state.get("reviewer_questions_to_researcher", [])

    hypotheses = state["synthesizer_hypotheses"]
    insights = state["synthesizer_insights"]
    synthesizer_reasoning = state.get("synthesizer_reasoning", "")
    novel_connections = state.get("synthesizer_novel_connections", [])

    gaps = state["questioner_gaps"]
    questions = state["questioner_questions"]
    questioner_reasoning = state.get("questioner_reasoning", "")

    sources = state["researcher_sources"]
    agent_interactions = state.get("agent_interactions", [])

    system_prompt = """You are a research report compiler specializing in transparent, verifiable documentation.

YOUR MISSION:
Create a comprehensive research report that:
1. Presents findings with full reasoning traces
2. Shows how agents collaborated and challenged each other
3. Makes all conclusions verifiable (with citations)
4. Highlights novel contributions clearly
5. Maintains scientific rigor throughout

REPORT STRUCTURE:
1. Executive Summary
   - Key findings (with confidence levels)
   - Novel hypotheses generated
   - Critical gaps identified
   
2. Detailed Analysis
   - Findings with evidence and reasoning
   - Methodologies and their limitations
   
3. Critical Review
   - Strengths validated
   - Weaknesses identified
   - Alternative interpretations
   
4. Novel Insights & Hypotheses
   - Original contributions
   - Why they're novel
   - How to test them
   
5. Research Gaps & Questions
   - What's missing
   - Priority questions
   - Suggested approaches
   
6. Agent Collaboration Log
   - How agents interacted
   - Questions raised between agents
   - How critiques shaped the analysis
   
7. Verification & Sources
   - All citations
   - Confidence assessments
   - Reasoning traces

Make it clear, verifiable, and scientifically rigorous."""

    user_prompt = f"""Compile the comprehensive research report:

RESEARCHER'S FINDINGS ({len(findings)} total, avg confidence: {researcher_confidence:.1f}%):
{chr(10).join([f"- {f.get('finding', 'N/A')} [Conf: {f.get('confidence', 'N/A')}%]" for f in findings[:10]])}

REVIEWER'S CRITIQUE:
Strengths: {chr(10).join([s.get('claim', 'N/A') for s in strengths[:5]])}
Weaknesses: {chr(10).join([w.get('claim', 'N/A') for w in weaknesses[:5]])}
Questions Raised: {chr(10).join([f"- {q}" for q in reviewer_questions[:5]])}

SYNTHESIZER'S CONTRIBUTIONS:
Novel Hypotheses: {chr(10).join([h.get('statement', 'N/A') for h in hypotheses[:5]])}
Key Insights: {chr(10).join([i.get('statement', 'N/A') for i in insights[:5]])}
Novel Connections: {chr(10).join([f"- {nc.get('connection', 'N/A')}" for nc in novel_connections[:3]])}

QUESTIONER'S GAP ANALYSIS:
Critical Gaps: {chr(10).join([g.get('gap', 'N/A') for g in gaps[:5]])}
Research Questions: {chr(10).join([q.get('question', 'N/A') for q in questions[:7]])}

AGENT INTERACTIONS:
{chr(10).join([f"[{i.get('from', 'Agent')} → {i.get('to', 'Agent')}]: {i.get('content', 'N/A')}" for i in agent_interactions])}

REASONING TRACES:
Researcher: {researcher_reasoning[:300]}...
Reviewer: {reviewer_reasoning[:300]}...
Synthesizer: {synthesizer_reasoning[:300]}...
Questioner: {questioner_reasoning[:300]}...

SOURCES:
{chr(10).join([f"[{i+1}] {s.get('source', 'Unknown')}" for i, s in enumerate(sources[:15])])}

Compile a comprehensive report that shows the full reasoning process and agent collaboration."""

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

        # Create reasoning graph showing agent flow
        reasoning_graph = {
            "nodes": [
                {"id": "researcher", "label": "RESEARCHER", "findings": len(findings)},
                {"id": "reviewer", "label": "REVIEWER", "questions": len(reviewer_questions)},
                {"id": "synthesizer", "label": "SYNTHESIZER", "hypotheses": len(hypotheses)},
                {"id": "questioner", "label": "QUESTIONER", "gaps": len(gaps)}
            ],
            "edges": [
                {"from": "researcher", "to": "reviewer", "type": "analysis"},
                {"from": "reviewer", "to": "researcher", "type": "questions", "count": len(reviewer_questions)},
                {"from": "reviewer", "to": "synthesizer", "type": "critique"},
                {"from": "synthesizer", "to": "questioner", "type": "hypotheses"},
                {"from": "questioner", "to": "all", "type": "questions"}
            ],
            "interactions": agent_interactions
        }

        # Calculate verification scores
        verification_scores = {
            "average_confidence": researcher_confidence,
            "findings_with_evidence": len([f for f in findings if f.get('evidence')]),
            "total_findings": len(findings),
            "hypotheses_with_testability": len([h for h in hypotheses if h.get('testability')]),
            "total_hypotheses": len(hypotheses),
            "questions_with_approach": len([q for q in questions if q.get('approach')]),
            "total_questions": len(questions),
            "agent_interactions": len(agent_interactions),
            "novel_contributions": len(novel_connections)
        }

        logger.info("FORMATTER: Report compiled with full verification")

        return {
            "messages": [AIMessage(content="FORMATTER completed comprehensive report")],
            "final_report": report,
            "formatter_status": "success",
            "workflow_status": "success",
            "current_step": 5,
            "reasoning_graph": reasoning_graph,
            "verification_scores": verification_scores
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
        "error_message": "",
        "agent_interactions": [],
        "verification_scores": {}
    }

    print(f"\n{'=' * 70}")
    print("ENHANCED MULTI-AGENT RESEARCH LAB")
    print("With Reasoning Traces & Agent Collaboration")
    print(f"{'=' * 70}\n")

    # Run the graph
    result = graph.invoke(initial_state, config)

    if result["workflow_status"] == "success":
        print(f"\n{'=' * 70}")
        print("WORKFLOW COMPLETE")
        print(f"{'=' * 70}")

        # Print verification scores
        scores = result.get("verification_scores", {})
        print(f"\n📊 VERIFICATION METRICS:")
        print(f"  • Average Confidence: {scores.get('average_confidence', 0):.1f}%")
        print(f"  • Findings with Evidence: {scores.get('findings_with_evidence', 0)}/{scores.get('total_findings', 0)}")
        print(f"  • Testable Hypotheses: {scores.get('hypotheses_with_testability', 0)}/{scores.get('total_hypotheses', 0)}")
        print(f"  • Questions with Approach: {scores.get('questions_with_approach', 0)}/{scores.get('total_questions', 0)}")
        print(f"  • Agent Interactions: {scores.get('agent_interactions', 0)}")
        print(f"  • Novel Contributions: {scores.get('novel_contributions', 0)}")

        print(f"\n{'=' * 70}")
        print("FINAL REPORT")
        print(f"{'=' * 70}\n")
        print(result["final_report"])
        print(f"\n{'=' * 70}\n")

        # Print reasoning graph
        print(f"\n🔄 AGENT COLLABORATION FLOW:")
        graph_data = result.get("reasoning_graph", {})
        for interaction in graph_data.get("interactions", []):
            print(f"  [{interaction.get('from', 'Agent')}] → [{interaction.get('to', 'Agent')}]: {interaction.get('content', 'N/A')}")

    else:
        print(f"\n❌ WORKFLOW FAILED: {result.get('error_message', 'Unknown error')}\n")

    return result


def export_reasoning_graph(state: ResearchState, output_path: str = "reasoning_graph.json"):
    """Export the reasoning graph for visualization"""
    import json

    graph_data = state.get("reasoning_graph", {})

    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"✓ Reasoning graph exported to {output_path}")


if __name__ == "__main__":
    # Run analysis
    result = run_research_analysis()

    # Save report to file
    if result["workflow_status"] == "success":
        with open("research_report.txt", "w") as f:
            f.write("="*70 + "\n")
            f.write("ENHANCED MULTI-AGENT RESEARCH REPORT\n")
            f.write("="*70 + "\n\n")

            # Add verification section
            scores = result.get("verification_scores", {})
            f.write("VERIFICATION METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Average Confidence: {scores.get('average_confidence', 0):.1f}%\n")
            f.write(f"Findings with Evidence: {scores.get('findings_with_evidence', 0)}/{scores.get('total_findings', 0)}\n")
            f.write(f"Testable Hypotheses: {scores.get('hypotheses_with_testability', 0)}/{scores.get('total_hypotheses', 0)}\n")
            f.write(f"Novel Contributions: {scores.get('novel_contributions', 0)}\n\n")

            # Add agent collaboration section
            f.write("AGENT COLLABORATION LOG\n")
            f.write("-"*70 + "\n")
            for interaction in result.get("agent_interactions", []):
                f.write(f"[{interaction.get('from', 'Agent')}] → [{interaction.get('to', 'Agent')}]: {interaction.get('content', 'N/A')}\n")
            f.write("\n")

            f.write("="*70 + "\n\n")
            f.write(result["final_report"])

        print("✓ Report saved to enhanced_research_report.txt")

        # Export reasoning graph
        export_reasoning_graph(result)