from src.llm import get_llm, parse_llm_response
from src.prompts import RESUME_ANALYSIS_PROMPT


def analyze_resume(
    resume_context,
    job_description,
    retrieved_documents=None,
    resume_chunks=None
):
    """
    Analyze a resume against a job description using
    retrieved resume chunks and GPT-OSS-120B.
    """

    llm = get_llm()

    prompt = RESUME_ANALYSIS_PROMPT.invoke(
        {
            "resume_context": resume_context,
            "job_description": job_description
        }
    )

    response = llm.invoke(prompt)

    analysis = parse_llm_response(response)

    # Keep all pipeline information together.
    return {
        "analysis": analysis,

        # Documents retrieved from FAISS
        "retrieved_documents": retrieved_documents or [],

        # Resume chunks used to build the vector store
        "resume_chunks": resume_chunks or []
    }