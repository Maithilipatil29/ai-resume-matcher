import os
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


def get_llm():
    """
    Create and return the Groq LLM.
    """

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )


def parse_llm_response(response):
    """
    Convert the LLM response into a Python dictionary.

    Handles:
    - Plain JSON
    - JSON inside ```json ... ```
    - JSON surrounded by extra text
    """

    # LangChain AIMessage
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    content = content.strip()

    # Remove markdown code fences
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"^```\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    content = content.strip()

    # First attempt: entire response is JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Second attempt: find JSON object inside response
    match = re.search(
        r"\{.*\}",
        content,
        flags=re.DOTALL
    )

    if match:
        json_text = match.group(0)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # If JSON parsing fails
    return {
        "match_percentage": None,
        "matched_skills": [],
        "missing_skills": [],
        "experience_analysis": "",
        "project_analysis": "",
        "education_analysis": "",
        "evidence": [],
        "improvement_suggestions": [],
        "overall_summary": content,
        "parsing_error": True
    }