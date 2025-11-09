"""
Multi-Agent System for Research Analysis
Each agent has a specialized role in the research analysis pipeline.
"""

import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Handle both script execution and module import
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


class BaseAgent:
    """Base class for all research agents."""

    def __init__(
            self,
            name: str,
            role: str,
            rag_pipeline: RAGPipeline,
            temperature: float = 0.3,
            max_retrieval_docs: int = 8
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent name
            role: Agent's role description
            rag_pipeline: RAG pipeline for document retrieval
            temperature: LLM temperature (lower for more focused, accurate responses)
            max_retrieval_docs: Maximum documents to retrieve
        """
        self.name = name
        self.role = role
        self.rag_pipeline = rag_pipeline
        self.temperature = temperature
        self.max_retrieval_docs = max_retrieval_docs

        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize LLM with lower temperature for accuracy
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )

        logger.info(f"Initialized {self.name} agent (temperature={temperature})")

    def retrieve_context(self, query: str, k: Optional[int] = None) -> Dict:
        """
        Retrieve relevant context using RAG pipeline.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Dictionary with context and sources
        """
        if k is None:
            k = self.max_retrieval_docs

        result = self.rag_pipeline.answer_question(query, k=k, return_sources=True)
        return result

    def process(self, input_data: Dict, query: str) -> Dict:
        """
        Process input data and generate output.
        Must be implemented by subclass.

        Args:
            input_data: Input from previous agent
            query: Original research query

        Returns:
            Dictionary with agent's output
        """
        raise NotImplementedError("Subclass must implement process method")


class ResearcherAgent(BaseAgent):
    """Agent that analyzes research papers and extracts key findings."""

    def __init__(self, rag_pipeline: RAGPipeline, temperature: float = 0.2):
        super().__init__(
            name="RESEARCHER",
            role="Analyzes research papers, extracts key findings, methodologies, and conclusions",
            rag_pipeline=rag_pipeline,
            temperature=temperature,
            max_retrieval_docs=10
        )

    def process(self, input_data: Dict, query: str) -> Dict:
        """
        Analyze papers and extract key findings.

        Args:
            input_data: Initial query or research topic
            query: Research query/topic

        Returns:
            Dictionary with analysis findings
        """
        logger.info(f"{self.name}: Starting analysis of research papers...")

        # Retrieve relevant documents
        retrieval_query = query if isinstance(input_data, str) else input_data.get('query', query)
        context_result = self.retrieve_context(retrieval_query, k=10)

        if not context_result.get('sources'):
            return {
                "agent": self.name,
                "status": "error",
                "message": "No relevant documents found",
                "findings": [],
                "sources": []
            }

        # Create analysis prompt
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

        user_prompt = f"""Analyze the following research papers related to: {query}

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
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            analysis = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"{self.name}: Analysis complete")

            return {
                "agent": self.name,
                "status": "success",
                "analysis": analysis,
                "findings": self._extract_findings(analysis),
                "sources": sources,
                "num_sources": len(sources)
            }
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            return {
                "agent": self.name,
                "status": "error",
                "message": str(e),
                "findings": [],
                "sources": []
            }

    def _extract_findings(self, analysis: str) -> List[str]:
        """Extract key findings from analysis text."""
        # Simple extraction - can be enhanced
        lines = analysis.split('\n')
        findings = [line.strip() for line in lines if
                    line.strip() and (line.strip().startswith('-') or line.strip()[0].isdigit())]
        return findings[:10]  # Return top 10 findings


class ReviewerAgent(BaseAgent):
    """Agent that critiques findings and identifies strengths/weaknesses."""

    def __init__(self, rag_pipeline: RAGPipeline, temperature: float = 0.3):
        super().__init__(
            name="REVIEWER",
            role="Critiques research findings, identifies strengths, weaknesses, and potential biases",
            rag_pipeline=rag_pipeline,
            temperature=temperature,
            max_retrieval_docs=8
        )

    def process(self, input_data: Dict, query: str) -> Dict:
        """
        Critique the researcher's findings.

        Args:
            input_data: Output from ResearcherAgent
            query: Original research query

        Returns:
            Dictionary with critique
        """
        logger.info(f"{self.name}: Starting critique of findings...")

        if input_data.get('status') != 'success':
            return {
                "agent": self.name,
                "status": "error",
                "message": "Invalid input from previous agent",
                "critique": ""
            }

        researcher_analysis = input_data.get('analysis', '')
        findings = input_data.get('findings', [])
        sources = input_data.get('sources', [])

        # Retrieve additional context for critique
        critique_query = f"methodology limitations weaknesses {query}"
        context_result = self.retrieve_context(critique_query, k=6)

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

        user_prompt = f"""Review and critique the following research analysis related to: {query}

RESEARCHER'S ANALYSIS:
{researcher_analysis}

KEY FINDINGS:
{chr(10).join([f"- {f}" for f in findings[:10]])}

ADDITIONAL CONTEXT:
{context_result.get('answer', 'No additional context available')}

Please provide a thorough critique focusing on:
1. Strengths of the research and analysis
2. Weaknesses or limitations
3. Potential biases or methodological concerns
4. Gaps in the analysis or missing information
5. Consistency and logical coherence

Remember: Base your critique on the actual content. Do not invent criticisms."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            critique = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"{self.name}: Critique complete")

            return {
                "agent": self.name,
                "status": "success",
                "critique": critique,
                "researcher_analysis": researcher_analysis,
                "strengths": self._extract_section(critique, "strengths"),
                "weaknesses": self._extract_section(critique, "weaknesses"),
                "sources": sources + context_result.get('sources', [])
            }
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            return {
                "agent": self.name,
                "status": "error",
                "message": str(e),
                "critique": ""
            }

    def _extract_section(self, text: str, section: str) -> List[str]:
        """Extract a specific section from critique text."""
        lines = text.split('\n')
        section_lines = []
        in_section = False
        for line in lines:
            if section.lower() in line.lower() and ('-' in line or line[0].isdigit() if line.strip() else False):
                in_section = True
            if in_section:
                if line.strip() and (line.strip().startswith('-') or (line.strip() and line.strip()[0].isdigit())):
                    section_lines.append(line.strip())
                elif line.strip() and not line.strip()[0].isdigit() and '-' not in line and len(line.strip()) > 50:
                    break
        return section_lines[:5]


class SynthesizerAgent(BaseAgent):
    """Agent that synthesizes insights and generates hypotheses."""

    def __init__(self, rag_pipeline: RAGPipeline, temperature: float = 0.4):
        super().__init__(
            name="SYNTHESIZER",
            role="Synthesizes findings and critiques to generate new insights and testable hypotheses",
            rag_pipeline=rag_pipeline,
            temperature=temperature,
            max_retrieval_docs=8
        )

    def process(self, input_data: Dict, query: str) -> Dict:
        """
        Synthesize findings and generate hypotheses.

        Args:
            input_data: Output from ReviewerAgent
            query: Original research query

        Returns:
            Dictionary with synthesis and hypotheses
        """
        logger.info(f"{self.name}: Starting synthesis...")

        if input_data.get('status') != 'success':
            return {
                "agent": self.name,
                "status": "error",
                "message": "Invalid input from previous agent",
                "synthesis": "",
                "hypotheses": []
            }

        researcher_analysis = input_data.get('researcher_analysis', '')
        critique = input_data.get('critique', '')
        strengths = input_data.get('strengths', [])
        weaknesses = input_data.get('weaknesses', [])

        # Retrieve context for synthesis
        synthesis_query = f"hypotheses research questions future work {query}"
        context_result = self.retrieve_context(synthesis_query, k=6)

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

        user_prompt = f"""Synthesize the following research analysis and critique related to: {query}

RESEARCHER'S FINDINGS:
{researcher_analysis}

REVIEWER'S CRITIQUE:
{critique}

STRENGTHS IDENTIFIED:
{chr(10).join([f"- {s}" for s in strengths[:5]])}

WEAKNESSES IDENTIFIED:
{chr(10).join([f"- {w}" for w in weaknesses[:5]])}

ADDITIONAL CONTEXT:
{context_result.get('answer', 'No additional context available')}

Please synthesize this information to:
1. Identify key insights and patterns
2. Connect findings from different sources
3. Generate 3-5 testable hypotheses
4. Propose specific research directions
5. Explain the evidence base for each hypothesis

Remember: Hypotheses must be grounded in the actual findings. Be specific and testable."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            synthesis = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"{self.name}: Synthesis complete")

            return {
                "agent": self.name,
                "status": "success",
                "synthesis": synthesis,
                "hypotheses": self._extract_hypotheses(synthesis),
                "insights": self._extract_insights(synthesis),
                "researcher_analysis": researcher_analysis,
                "critique": critique
            }
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            return {
                "agent": self.name,
                "status": "error",
                "message": str(e),
                "synthesis": "",
                "hypotheses": []
            }

    def _extract_hypotheses(self, synthesis: str) -> List[str]:
        """Extract hypotheses from synthesis text."""
        hypotheses = []
        lines = synthesis.split('\n')
        for line in lines:
            if 'hypothesis' in line.lower() or 'h1' in line.lower() or 'h2' in line.lower() or 'h3' in line.lower():
                if line.strip() and len(line.strip()) > 20:
                    hypotheses.append(line.strip())
        return hypotheses[:5]

    def _extract_insights(self, synthesis: str) -> List[str]:
        """Extract key insights from synthesis text."""
        insights = []
        lines = synthesis.split('\n')
        for line in lines:
            if (
                    'insight' in line.lower() or 'pattern' in line.lower() or 'relationship' in line.lower()) and line.strip():
                if len(line.strip()) > 30:
                    insights.append(line.strip())
        return insights[:5]


class QuestionerAgent(BaseAgent):
    """Agent that identifies gaps and generates follow-up questions."""

    def __init__(self, rag_pipeline: RAGPipeline, temperature: float = 0.4):
        super().__init__(
            name="QUESTIONER",
            role="Identifies research gaps and generates critical follow-up questions",
            rag_pipeline=rag_pipeline,
            temperature=temperature,
            max_retrieval_docs=6
        )

    def process(self, input_data: Dict, query: str) -> Dict:
        """
        Identify gaps and generate questions.

        Args:
            input_data: Output from SynthesizerAgent
            query: Original research query

        Returns:
            Dictionary with gaps and questions
        """
        logger.info(f"{self.name}: Identifying gaps and generating questions...")

        if input_data.get('status') != 'success':
            return {
                "agent": self.name,
                "status": "error",
                "message": "Invalid input from previous agent",
                "gaps": [],
                "questions": []
            }

        synthesis = input_data.get('synthesis', '')
        hypotheses = input_data.get('hypotheses', [])
        insights = input_data.get('insights', [])
        researcher_analysis = input_data.get('researcher_analysis', '')
        critique = input_data.get('critique', '')

        # Retrieve context for gap identification
        gap_query = f"research gaps limitations future work {query}"
        context_result = self.retrieve_context(gap_query, k=5)

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

        user_prompt = f"""Identify gaps and generate questions based on the following research analysis related to: {query}

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
{context_result.get('answer', 'No additional context available')}

Please identify:
1. Knowledge gaps in the current research
2. Unanswered questions that emerged
3. Critical follow-up questions (5-7 questions)
4. Research priorities for future work
5. Areas needing further investigation

Remember: Questions should be specific and answerable. Base gaps on actual limitations identified."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            gap_analysis = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"{self.name}: Gap analysis complete")

            return {
                "agent": self.name,
                "status": "success",
                "gap_analysis": gap_analysis,
                "gaps": self._extract_gaps(gap_analysis),
                "questions": self._extract_questions(gap_analysis),
                "synthesis": synthesis,
                "hypotheses": hypotheses
            }
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            return {
                "agent": self.name,
                "status": "error",
                "message": str(e),
                "gaps": [],
                "questions": []
            }

    def _extract_gaps(self, text: str) -> List[str]:
        """Extract knowledge gaps from text."""
        gaps = []
        lines = text.split('\n')
        for line in lines:
            if 'gap' in line.lower() and line.strip() and len(line.strip()) > 20:
                gaps.append(line.strip())
        return gaps[:5]

    def _extract_questions(self, text: str) -> List[str]:
        """Extract questions from text."""
        questions = []
        lines = text.split('\n')
        for line in lines:
            if '?' in line and line.strip() and len(line.strip()) > 10:
                questions.append(line.strip())
        return questions[:7]


class FormatterAgent(BaseAgent):
    """Agent that compiles the final research report."""

    def __init__(self, rag_pipeline: RAGPipeline, temperature: float = 0.3):
        super().__init__(
            name="FORMATTER",
            role="Compiles a comprehensive research report from all agent outputs",
            rag_pipeline=rag_pipeline,
            temperature=temperature,
            max_retrieval_docs=0  # Formatter doesn't need to retrieve new docs
        )

    def process(self, input_data: Dict, query: str) -> Dict:
        """
        Compile final report from all agent outputs.

        Args:
            input_data: Complete workflow data from all agents
            query: Original research query

        Returns:
            Dictionary with formatted report
        """
        logger.info(f"{self.name}: Compiling final report...")

        # Extract data from all agents
        researcher_data = input_data.get('researcher', {})
        reviewer_data = input_data.get('reviewer', {})
        synthesizer_data = input_data.get('synthesizer', {})
        questioner_data = input_data.get('questioner', {})

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

        user_prompt = f"""Compile a comprehensive research report for: {query}

RESEARCHER'S ANALYSIS:
{researcher_data.get('analysis', 'N/A')}

REVIEWER'S CRITIQUE:
{reviewer_data.get('critique', 'N/A')}

SYNTHESIZER'S SYNTHESIS:
{synthesizer_data.get('synthesis', 'N/A')}

HYPOTHESES:
{chr(10).join([f"- {h}" for h in synthesizer_data.get('hypotheses', [])])}

QUESTIONER's GAP ANALYSIS:
{questioner_data.get('gap_analysis', 'N/A')}

RESEARCH QUESTIONS:
{chr(10).join([f"- {q}" for q in questioner_data.get('questions', [])])}

SOURCES:
{chr(10).join([f"- {s.get('source', 'Unknown')}" for s in researcher_data.get('sources', [])[:10]])}

Please compile a comprehensive research report with proper structure, citations, and all key information from the analyses above."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            report = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"{self.name}: Report compiled")

            return {
                "agent": self.name,
                "status": "success",
                "report": report,
                "query": query,
                "researcher": researcher_data,
                "reviewer": reviewer_data,
                "synthesizer": synthesizer_data,
                "questioner": questioner_data,
                "sources": researcher_data.get('sources', [])
            }
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            return {
                "agent": self.name,
                "status": "error",
                "message": str(e),
                "report": ""
            }
