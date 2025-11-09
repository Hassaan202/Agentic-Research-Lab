"""
Summarizer Agent Module
Generates structured summaries of research papers using Google Gemini.
"""

import json
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizerAgent:
    """
    Agent that summarizes research papers into structured metadata.

    Uses Google Gemini to extract key information from papers and
    structure it in a standardized JSON format.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize the summarizer agent.

        Args:
            model_name: Google Gemini model to use for summarization
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please create a .env file with your Google API key."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )

        self.prompt = PromptTemplate(
            input_variables=["paper_text"],
            template="""You are a research paper analyzer. Extract and summarize the key information from the paper below.

Paper Text:
--------------------
{paper_text}
--------------------

Analyze this paper and return ONLY valid JSON with this exact structure:
{{
    "title": "paper title",
    "authors": "list of authors",
    "year": "publication year",
    "research_question": "main research question or objective",
    "methods": "research methods and approach used",
    "key_findings": "main findings and results",
    "limitations": "limitations and future work mentioned"
}}

Important: Return ONLY the JSON object, no additional text or markdown formatting."""
        )

    def summarize_paper(self, paper_text: str, source_file: str = "") -> Dict[str, Any]:
        """
        Generate structured summary of a research paper.

        Args:
            paper_text: Full text of the paper
            source_file: Source filename for metadata

        Returns:
            Dictionary containing structured summary
        """
        try:
            # Truncate very long papers to fit context window
            max_chars = 1000000  # Adjust based on model limits
            if len(paper_text) > max_chars:
                logger.warning(f"Paper too long ({len(paper_text)} chars), truncating to {max_chars}")
                paper_text = paper_text[:max_chars]

            prompt = self.prompt.format(paper_text=paper_text)
            response = self.llm.invoke(prompt)

            # Parse response
            response_text = response.content.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            summary = json.loads(response_text)

            # Add metadata
            summary["source_file"] = source_file
            summary["summary_generated"] = True

            logger.info(f"Generated summary for: {summary.get('title', 'Unknown')}")
            return summary

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {
                "title": "Parse Error",
                "authors": "",
                "year": "",
                "research_question": "",
                "methods": "",
                "key_findings": "",
                "limitations": "",
                "source_file": source_file,
                "summary_generated": False,
                "raw_output": response.content if 'response' in locals() else ""
            }
        except Exception as e:
            logger.error(f"Error summarizing paper: {str(e)}")
            return {
                "title": "Error",
                "authors": "",
                "year": "",
                "research_question": "",
                "methods": "",
                "key_findings": "",
                "limitations": "",
                "source_file": source_file,
                "summary_generated": False,
                "error": str(e)
            }