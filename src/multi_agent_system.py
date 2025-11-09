"""
Multi-Agent Research System Orchestrator
Manages the workflow: RESEARCHER → REVIEWER → SYNTHESIZER → QUESTIONER → FORMATTER
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Handle both script execution and module import
try:
    from .rag_pipeline import RAGPipeline
    from .agents import (
        ResearcherAgent,
        ReviewerAgent,
        SynthesizerAgent,
        QuestionerAgent,
        FormatterAgent
    )
except ImportError:
    # For script execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from rag_pipeline import RAGPipeline
    from agents import (
        ResearcherAgent,
        ReviewerAgent,
        SynthesizerAgent,
        QuestionerAgent,
        FormatterAgent
    )

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAgentResearchSystem:
    """
    Orchestrates multiple research agents in a sequential workflow.
    
    Workflow:
    START → RESEARCHER → REVIEWER → SYNTHESIZER → QUESTIONER → FORMATTER → END
    """
    
    def __init__(
        self,
        vector_db_path: str = "vector_db",
        collection_name: str = "research_documents",
        llm_model: str = "gemini-2.5-flash",
        temperature: float = 0.3
    ):
        """
        Initialize the multi-agent research system.
        
        Args:
            vector_db_path: Path to vector database
            collection_name: Name of ChromaDB collection
            llm_model: LLM model name
            temperature: Base temperature for agents (individual agents may override)
        """
        logger.info("Initializing Multi-Agent Research System...")
        
        # Initialize RAG pipeline (shared by all agents)
        self.rag_pipeline = RAGPipeline(
            vector_db_path=vector_db_path,
            collection_name=collection_name,
            llm_model=llm_model,
            temperature=temperature,
            max_retrieval_docs=8
        )
        
        # Initialize all agents
        logger.info("Initializing agents...")
        self.researcher = ResearcherAgent(
            rag_pipeline=self.rag_pipeline,
            temperature=0.2  # Lower temperature for factual accuracy
        )
        self.reviewer = ReviewerAgent(
            rag_pipeline=self.rag_pipeline,
            temperature=0.3  # Slightly higher for critical thinking
        )
        self.synthesizer = SynthesizerAgent(
            rag_pipeline=self.rag_pipeline,
            temperature=0.4  # Higher for creative synthesis
        )
        self.questioner = QuestionerAgent(
            rag_pipeline=self.rag_pipeline,
            temperature=0.4  # Higher for generating questions
        )
        self.formatter = FormatterAgent(
            rag_pipeline=self.rag_pipeline,
            temperature=0.3  # Balanced for clear formatting
        )
        
        logger.info("Multi-Agent Research System initialized successfully")
    
    def run_research_workflow(self, query: str, verbose: bool = True) -> Dict:
        """
        Execute the complete research workflow.
        
        Args:
            query: Research query or topic
            verbose: Whether to print progress
            
        Returns:
            Dictionary with complete workflow results
        """
        if verbose:
            print("\n" + "="*70)
            print("MULTI-AGENT RESEARCH SYSTEM")
            print("="*70)
            print(f"Research Query: {query}")
            print("="*70)
            print("\nWorkflow: START → RESEARCHER → REVIEWER → SYNTHESIZER → QUESTIONER → FORMATTER → END\n")
        
        workflow_data = {
            "query": query,
            "workflow": [],
            "researcher": {},
            "reviewer": {},
            "synthesizer": {},
            "questioner": {},
            "formatter": {},
            "status": "in_progress"
        }
        
        try:
            # STEP 1: RESEARCHER - Analyze papers
            if verbose:
                print("[1/5] RESEARCHER: Analyzing research papers...")
            researcher_output = self.researcher.process(
                input_data={"query": query},
                query=query
            )
            workflow_data["researcher"] = researcher_output
            workflow_data["workflow"].append({
                "step": 1,
                "agent": "RESEARCHER",
                "status": researcher_output.get("status", "unknown")
            })
            
            if researcher_output.get("status") != "success":
                workflow_data["status"] = "error"
                workflow_data["error"] = f"Researcher agent failed: {researcher_output.get('message', 'Unknown error')}"
                if verbose:
                    print(f"❌ ERROR: {workflow_data['error']}")
                return workflow_data
            
            if verbose:
                print(f"✓ RESEARCHER: Found {researcher_output.get('num_sources', 0)} sources")
            
            # STEP 2: REVIEWER - Critique findings
            if verbose:
                print("[2/5] REVIEWER: Critiquing findings...")
            reviewer_output = self.reviewer.process(
                input_data=researcher_output,
                query=query
            )
            workflow_data["reviewer"] = reviewer_output
            workflow_data["workflow"].append({
                "step": 2,
                "agent": "REVIEWER",
                "status": reviewer_output.get("status", "unknown")
            })
            
            if reviewer_output.get("status") != "success":
                workflow_data["status"] = "error"
                workflow_data["error"] = f"Reviewer agent failed: {reviewer_output.get('message', 'Unknown error')}"
                if verbose:
                    print(f"❌ ERROR: {workflow_data['error']}")
                return workflow_data
            
            if verbose:
                print("✓ REVIEWER: Critique complete")
            
            # STEP 3: SYNTHESIZER - Generate hypotheses
            if verbose:
                print("[3/5] SYNTHESIZER: Synthesizing insights and generating hypotheses...")
            synthesizer_output = self.synthesizer.process(
                input_data=reviewer_output,
                query=query
            )
            workflow_data["synthesizer"] = synthesizer_output
            workflow_data["workflow"].append({
                "step": 3,
                "agent": "SYNTHESIZER",
                "status": synthesizer_output.get("status", "unknown")
            })
            
            if synthesizer_output.get("status") != "success":
                workflow_data["status"] = "error"
                workflow_data["error"] = f"Synthesizer agent failed: {synthesizer_output.get('message', 'Unknown error')}"
                if verbose:
                    print(f"❌ ERROR: {workflow_data['error']}")
                return workflow_data
            
            hypotheses_count = len(synthesizer_output.get("hypotheses", []))
            if verbose:
                print(f"✓ SYNTHESIZER: Generated {hypotheses_count} hypotheses")
            
            # STEP 4: QUESTIONER - Identify gaps
            if verbose:
                print("[4/5] QUESTIONER: Identifying research gaps and generating questions...")
            questioner_output = self.questioner.process(
                input_data=synthesizer_output,
                query=query
            )
            workflow_data["questioner"] = questioner_output
            workflow_data["workflow"].append({
                "step": 4,
                "agent": "QUESTIONER",
                "status": questioner_output.get("status", "unknown")
            })
            
            if questioner_output.get("status") != "success":
                workflow_data["status"] = "error"
                workflow_data["error"] = f"Questioner agent failed: {questioner_output.get('message', 'Unknown error')}"
                if verbose:
                    print(f"❌ ERROR: {workflow_data['error']}")
                return workflow_data
            
            questions_count = len(questioner_output.get("questions", []))
            if verbose:
                print(f"✓ QUESTIONER: Identified {questions_count} research questions")
            
            # STEP 5: FORMATTER - Compile report
            if verbose:
                print("[5/5] FORMATTER: Compiling final research report...")
            formatter_output = self.formatter.process(
                input_data=workflow_data,
                query=query
            )
            workflow_data["formatter"] = formatter_output
            workflow_data["workflow"].append({
                "step": 5,
                "agent": "FORMATTER",
                "status": formatter_output.get("status", "unknown")
            })
            
            if formatter_output.get("status") != "success":
                workflow_data["status"] = "error"
                workflow_data["error"] = f"Formatter agent failed: {formatter_output.get('message', 'Unknown error')}"
                if verbose:
                    print(f"❌ ERROR: {workflow_data['error']}")
                return workflow_data
            
            workflow_data["status"] = "success"
            workflow_data["report"] = formatter_output.get("report", "")
            
            if verbose:
                print("✓ FORMATTER: Report compiled")
                print("\n" + "="*70)
                print("WORKFLOW COMPLETE")
                print("="*70)
            
            return workflow_data
            
        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            workflow_data["status"] = "error"
            workflow_data["error"] = str(e)
            if verbose:
                print(f"❌ WORKFLOW ERROR: {str(e)}")
            return workflow_data
    
    def get_workflow_summary(self, workflow_data: Dict) -> str:
        """
        Generate a summary of the workflow execution.
        
        Args:
            workflow_data: Output from run_research_workflow
            
        Returns:
            Formatted summary string
        """
        if workflow_data.get("status") != "success":
            return f"Workflow failed: {workflow_data.get('error', 'Unknown error')}"
        
        summary = f"""
WORKFLOW SUMMARY
================
Query: {workflow_data.get('query', 'N/A')}
Status: {workflow_data.get('status', 'N/A')}

Agent Results:
- RESEARCHER: {workflow_data.get('researcher', {}).get('status', 'N/A')} ({workflow_data.get('researcher', {}).get('num_sources', 0)} sources)
- REVIEWER: {workflow_data.get('reviewer', {}).get('status', 'N/A')}
- SYNTHESIZER: {workflow_data.get('synthesizer', {}).get('status', 'N/A')} ({len(workflow_data.get('synthesizer', {}).get('hypotheses', []))} hypotheses)
- QUESTIONER: {workflow_data.get('questioner', {}).get('status', 'N/A')} ({len(workflow_data.get('questioner', {}).get('questions', []))} questions)
- FORMATTER: {workflow_data.get('formatter', {}).get('status', 'N/A')}
"""
        return summary
    
    def save_report(self, workflow_data: Dict, output_path: str = "research_report.txt"):
        """
        Save the research report to a file.
        
        Args:
            workflow_data: Output from run_research_workflow
            output_path: Path to save the report
        """
        if workflow_data.get("status") != "success":
            logger.warning("Cannot save report: workflow did not complete successfully")
            return
        
        report = workflow_data.get("report", "")
        if not report:
            logger.warning("No report to save")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("RESEARCH REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Research Query: {workflow_data.get('query', 'N/A')}\n\n")
                f.write("="*70 + "\n\n")
                f.write(report)
                f.write("\n\n" + "="*70 + "\n")
                f.write("SOURCES\n")
                f.write("="*70 + "\n\n")
                sources = workflow_data.get('researcher', {}).get('sources', [])
                for i, source in enumerate(sources, 1):
                    f.write(f"{i}. {source.get('source', 'Unknown')}\n")
                    if source.get('page'):
                        f.write(f"   Page: {source['page']}\n")
                    f.write("\n")
            
            logger.info(f"Report saved to {output_path}")
            print(f"✓ Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            print(f"❌ Error saving report: {str(e)}")


def main():
    """Main function to run the multi-agent research system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System - Analyze research papers with specialized agents"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Research query or topic to analyze"
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        default="vector_db",
        help="Path to vector database"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="research_documents",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research_report.txt",
        help="Output file path for report"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save report to file"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY not found. Please create a .env file with your Google API key. "
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
        sys.exit(1)
    
    # Initialize system
    try:
        system = MultiAgentResearchSystem(
            vector_db_path=args.vector_db,
            collection_name=args.collection,
            llm_model=args.model,
            temperature=args.temperature
        )
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        sys.exit(1)
    
    # Run workflow
    args.query = 'provide a comprehensive analysis of the documents'
    workflow_data = system.run_research_workflow(args.query, verbose=True)
    
    # Print summary
    if workflow_data.get("status") == "success":
        print(system.get_workflow_summary(workflow_data))
        
        # Print report
        report = workflow_data.get("report", "")
        if report:
            print("\n" + "="*70)
            print("RESEARCH REPORT")
            print("="*70)
            print(report)
            print("="*70 + "\n")
        
        # Save report
        if not args.no_save:
            system.save_report(workflow_data, args.output)
    else:
        print(f"\n❌ Workflow failed: {workflow_data.get('error', 'Unknown error')}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

