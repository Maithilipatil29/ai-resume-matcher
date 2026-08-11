import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.document_loader import load_document
from src.analyzer import analyze_resume


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 AI-Powered Resume & Job Description Matcher"
)

st.write(
    """
    Upload your resume and paste a job description.

    The system uses:

    **PDF/DOCX → NLP preprocessing → Chunking → "
    "Hugging Face Embeddings → FAISS → Semantic Retrieval → "
    "RAG → GPT-OSS-120B → Resume Analysis**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

# with st.sidebar:

#     st.header("⚙️ Pipeline")

#     st.write("📄 Document Parser")
#     st.write("🧹 Text Processing")
#     st.write("✂️ Text Chunking")
#     st.write("🤗 Sentence Transformers")
#     st.write("🔎 FAISS Vector Search")
#     st.write("📚 RAG")
#     st.write("🦜 LangChain")
#     st.write("🧠 GPT-OSS-120B")
#     st.write("📊 Streamlit")


# ============================================================
# RESUME UPLOAD
# ============================================================

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "Paste Job Description",
    height=300,
    placeholder="""
Example:

We are looking for an AI Engineer with experience in
Python, machine learning, LLMs, RAG, LangChain,
FastAPI, Docker and AWS.
"""
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🚀 Analyze Resume",
    type="primary"
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if resume_file is None:

        st.error(
            "Please upload a resume."
        )

        st.stop()


    if not job_description.strip():

        st.error(
            "Please enter a job description."
        )

        st.stop()


    # ========================================================
    # STEP 1 — LOAD RESUME
    # ========================================================

    with st.spinner(
        "📄 Reading resume..."
    ):

        try:

            resume_bytes = (
                resume_file.getvalue()
            )

            resume_text = load_document(
                resume_bytes,
                resume_file.name
            )

        except Exception as error:

            st.error(
                f"Resume loading failed: {error}"
            )

            st.stop()


    if not resume_text.strip():

        st.error(
            "Could not extract text from the resume."
        )

        st.stop()


    # ========================================================
    # STEP 2 — CREATE LANGCHAIN DOCUMENT
    # ========================================================

    with st.spinner(
        "📝 Preparing resume text..."
    ):

        documents = [
            Document(
                page_content=resume_text,
                metadata={
                    "source": resume_file.name
                }
            )
        ]


    # ========================================================
    # STEP 3 — TEXT CHUNKING
    # ========================================================

    with st.spinner(
        "✂️ Splitting resume into chunks..."
    ):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        resume_chunks = (
            text_splitter.split_documents(
                documents
            )
        )


    if not resume_chunks:

        st.error(
            "No resume chunks were created."
        )

        st.stop()


    # ========================================================
    # STEP 4 — HUGGING FACE EMBEDDINGS
    # ========================================================

    with st.spinner(
        "🤗 Creating resume embeddings..."
    ):

        try:

            embedding_model = (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )

            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model
            )

        except Exception as error:

            st.error(
                f"Embedding model failed: {error}"
            )

            st.stop()


    # ========================================================
    # STEP 5 — CREATE FAISS VECTOR DATABASE
    # ========================================================

    with st.spinner(
        "🔎 Building FAISS vector database..."
    ):

        try:

            vector_store = (
                FAISS.from_documents(
                    resume_chunks,
                    embeddings
                )
            )

        except Exception as error:

            st.error(
                f"FAISS vector database failed: {error}"
            )

            st.stop()


    # ========================================================
    # STEP 6 — RETRIEVAL
    # ========================================================

    with st.spinner(
        "🔍 Finding relevant resume information..."
    ):

        try:

            top_k = 5

            retrieved_documents = (
                vector_store.similarity_search_with_score(
                    job_description,
                    k=top_k
                )
            )

        except Exception as error:

            st.error(
                f"FAISS retrieval failed: {error}"
            )

            st.stop()


    if not retrieved_documents:

        st.error(
            "No relevant resume information was retrieved."
        )

        st.stop()


    # ========================================================
    # STEP 7 — BUILD RAG CONTEXT
    # ========================================================

    with st.spinner(
        "📚 Building RAG context..."
    ):

        resume_context_parts = []

        for document, score in retrieved_documents:

            resume_context_parts.append(
                document.page_content
            )


        resume_context = "\n\n".join(
            resume_context_parts
        )


    # ========================================================
    # STEP 8 — LLM ANALYSIS
    # ========================================================

    with st.spinner(
        "🧠 GPT-OSS-120B is analyzing the resume..."
    ):

        try:

            result = analyze_resume(
                resume_context=resume_context,
                job_description=job_description,
                retrieved_documents=retrieved_documents,
                resume_chunks=resume_chunks
            )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

            st.stop()


    # ========================================================
    # STEP 9 — EXTRACT RESULTS
    # ========================================================

    analysis = result.get(
        "analysis",
        {}
    )

    retrieved_documents = result.get(
        "retrieved_documents",
        []
    )

    resume_chunks = result.get(
        "resume_chunks",
        []
    )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.header(
        "📊 Resume Analysis"
    )


    # ========================================================
    # MATCH PERCENTAGE
    # ========================================================

    match_percentage = analysis.get(
        "match_percentage"
    )


    if match_percentage is not None:

        try:

            match_percentage = float(
                match_percentage
            )

            st.metric(
                "🎯 Resume Match",
                f"{match_percentage:.0f}%"
            )

        except (
            ValueError,
            TypeError
        ):

            st.info(
                f"Match Percentage: "
                f"{match_percentage}"
            )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    st.subheader(
        "📝 Overall Analysis"
    )

    st.write(
        analysis.get(
            "overall_summary",
            "No summary available."
        )
    )


    # ========================================================
    # MATCHED SKILLS
    # ========================================================

    st.subheader(
        "✅ Matched Skills"
    )

    matched_skills = analysis.get(
        "matched_skills",
        []
    )


    if matched_skills:

        for skill in matched_skills:

            # ----------------------------------------------
            # Dictionary format
            # ----------------------------------------------

            if isinstance(
                skill,
                dict
            ):

                skill_name = skill.get(
                    "skill",
                    ""
                )

                evidence = skill.get(
                    "evidence",
                    ""
                )

            # ----------------------------------------------
            # String format
            # ----------------------------------------------

            else:

                skill_name = str(
                    skill
                )

                evidence = ""


            if skill_name:

                st.write(
                    f"✅ **{skill_name}**"
                )


            if evidence:

                st.caption(
                    evidence
                )


    else:

        st.write(
            "No matched skills returned."
        )


    # ========================================================
    # MISSING SKILLS
    # ========================================================

    st.subheader(
        "❌ Missing Skills"
    )

    missing_skills = analysis.get(
        "missing_skills",
        []
    )


    if missing_skills:

        for skill in missing_skills:

            # Dictionary format

            if isinstance(
                skill,
                dict
            ):

                skill_name = skill.get(
                    "skill",
                    ""
                )

                importance = skill.get(
                    "importance",
                    "medium"
                )

                st.warning(
                    f"{skill_name} "
                    f"(Importance: {importance})"
                )

            # String format

            else:

                st.warning(
                    str(skill)
                )


    else:

        st.write(
            "No missing skills identified."
        )


    # ========================================================
    # EXPERIENCE ANALYSIS
    # ========================================================

    st.subheader(
        "💼 Experience Analysis"
    )

    st.write(
        analysis.get(
            "experience_analysis",
            "No experience analysis available."
        )
    )


    # ========================================================
    # PROJECT ANALYSIS
    # ========================================================

    st.subheader(
        "🚀 Project Analysis"
    )

    st.write(
        analysis.get(
            "project_analysis",
            "No project analysis available."
        )
    )


    # ========================================================
    # EDUCATION ANALYSIS
    # ========================================================

    st.subheader(
        "🎓 Education Analysis"
    )

    st.write(
        analysis.get(
            "education_analysis",
            "No education analysis available."
        )
    )


    # ========================================================
    # IMPROVEMENT SUGGESTIONS
    # ========================================================

    st.subheader(
        "💡 Improvement Suggestions"
    )

    suggestions = analysis.get(
        "improvement_suggestions",
        []
    )


    if suggestions:

        for suggestion in suggestions:

            st.write(
                f"• {suggestion}"
            )

    else:

        st.write(
            "No improvement suggestions returned."
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    st.subheader(
        "🔎 Retrieved Resume Evidence"
    )

    if retrieved_documents:

        for index, item in enumerate(
            retrieved_documents
        ):

            # similarity_search_with_score returns:
            #
            # (Document, score)

            if isinstance(
                item,
                tuple
            ):

                document = item[0]
                score = item[1]

            else:

                document = item
                score = None


            if score is not None:

                title = (
                    f"Chunk {index + 1} "
                    f"| Distance: {score:.4f}"
                )

            else:

                title = (
                    f"Chunk {index + 1}"
                )


            with st.expander(
                title
            ):

                if hasattr(
                    document,
                    "page_content"
                ):

                    st.write(
                        document.page_content
                    )

                else:

                    st.write(
                        str(document)
                    )


                if hasattr(
                    document,
                    "metadata"
                ):

                    metadata = (
                        document.metadata
                    )

                    if metadata:

                        st.caption(
                            f"Metadata: {metadata}"
                        )


    else:

        st.write(
            "No retrieved documents."
        )


    # ========================================================
    # PIPELINE INFORMATION
    # ========================================================

    with st.expander(
        "🔧 Pipeline Information"
    ):

        st.write(
            f"📄 Resume file: "
            f"{resume_file.name}"
        )

        st.write(
            f"📑 Resume chunks created: "
            f"{len(resume_chunks)}"
        )

        st.write(
            f"🔎 Retrieved chunks: "
            f"{len(retrieved_documents)}"
        )

        st.write(
            "🤗 Embedding model: "
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        st.write(
            "🗄️ Vector database: FAISS"
        )

        st.write(
            "🔍 Retrieval: Semantic similarity"
        )

        st.write(
            "🦜 Framework: LangChain"
        )

        st.write(
            "🧠 LLM: GPT-OSS-120B via Groq"
        )

        st.write(
            "📚 Architecture: RAG"
        )