import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.document_loader import load_document
from src.analyzer import analyze_resume

import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📑",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📑 AI-Powered Resume & Job Description Matcher"
)

st.write(
    """
Upload your resume and paste a job description.


"""
)
# The system uses:

# PDF/DOCX
# → Text Processing
# → Chunking
# → Hugging Face Embeddings
# → FAISS
# → Semantic Retrieval
# → RAG Context
# → GPT-OSS-120B
# → Resume Analysis

# ============================================================
# RESUME UPLOAD
# ============================================================

resume_file = st.file_uploader(
    "Upload Resume",
    type=[
        "pdf",
        "docx"
    ]
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

    # ========================================================
    # VALIDATION
    # ========================================================

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

        text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
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

            embeddings = (
                HuggingFaceEmbeddings(
                    model_name=embedding_model
                )
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
    # STEP 6 — SEMANTIC RETRIEVAL
    # ========================================================

    with st.spinner(
        "🔍 Finding relevant resume information..."
    ):

        try:

            top_k = 5

            retrieved_documents = (
                vector_store
                .similarity_search_with_score(
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
    # STEP 7 — BUILD RETRIEVED CONTEXT
    # ========================================================

    with st.spinner(
        "📚 Building relevant resume context..."
    ):

        resume_context_parts = []

        for document, score in retrieved_documents:

            resume_context_parts.append(
                document.page_content
            )


        resume_context = (
            "\n\n".join(
                resume_context_parts
            )
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

                retrieved_documents=(
                    retrieved_documents
                ),

                resume_chunks=(
                    resume_chunks
                )
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

    # ============================================================
    # OVERALL MATCH SCORE
    # ============================================================

    st.subheader("🎯 Overall Match Score")

    match_percentage = analysis.get(
        "match_percentage",
        0
    )

    try:

        match_percentage = float(match_percentage)

        # Keep score between 0 and 100
        match_percentage = max(
            0,
            min(
                100,
                match_percentage
            )
        )

        # --------------------------------------------------------
        # Gauge color based on score
        # --------------------------------------------------------

        if match_percentage < 40:
            gauge_color = "#EF4444"

        elif match_percentage < 70:
            gauge_color = "#F59E0B"

        elif match_percentage < 85:
            gauge_color = "#7BE366"

        else:
            gauge_color = "#167F40"

        # --------------------------------------------------------
        # Create semicircular gauge
        # --------------------------------------------------------

        fig = go.Figure(
            go.Indicator(

                mode="gauge",

                value=match_percentage,

                gauge={
                    "shape": "angular",

                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#CBD5E1",
                        "tickfont": {
                            "size": 12,
                            "color": "#64748B"
                        }
                    },

                    "bar": {
                        "color": gauge_color,
                        "thickness": 0.28
                    },

                    "bgcolor": "#F1F5F9",

                    "borderwidth": 2,

                    "bordercolor": "#CBD5E1"
                }
            )
        )

        # --------------------------------------------------------
        # Add percentage inside gauge
        # --------------------------------------------------------

        fig.add_annotation(

            x=0.5,
            y=0.32,

            text=f"<b>{match_percentage:.0f}%</b>",

            showarrow=False,

            font={
                "size": 42,
                "color": "#172554",
                "family": "Arial"
            },

            xanchor="center",
            yanchor="middle"
        )

        # --------------------------------------------------------
        # Add label below percentage
        # --------------------------------------------------------

        if match_percentage >= 85:

            score_label = "Excellent Match"

        elif match_percentage >= 70:

            score_label = "Strong Match"

        elif match_percentage >= 40:

            score_label = "Moderate Match"

        else:

            score_label = "Low Match"

        fig.add_annotation(

            x=0.5,
            y=0.18,

            text=score_label,

            showarrow=False,

            font={
                "size": 18,
                "color": "#64748B",
                "family": "Arial"
            },

            xanchor="center",
            yanchor="middle"
        )

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------

        fig.update_layout(

            height=430,

            margin={
                "l": 30,
                "r": 30,
                "t": 10,
                "b": 10
            },

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"
        )

        # --------------------------------------------------------
        # Display gauge
        # --------------------------------------------------------

        col1, col2, col3 = st.columns(
            [1, 2, 1]
        )

        with col2:

            st.plotly_chart(
                fig,
                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )

    except (
        ValueError,
        TypeError
    ):

        st.info(
            f"Match Score: {match_percentage}"
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
            "No matched skills identified."
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

            else:

                st.warning(
                    str(skill)
                )

    else:

        st.success(
            "No major missing skills identified."
        )


    # ========================================================
    # EXPERIENCE ANALYSIS
    # ========================================================

    st.subheader(
        "💼 Experience Analysis"
    )

    experience_analysis = analysis.get(
        "experience_analysis",
        ""
    )


    if experience_analysis:

        st.write(
            experience_analysis
        )

    else:

        st.write(
            "No experience analysis available."
        )


    # ========================================================
    # PROJECT ANALYSIS
    # ========================================================

    st.subheader(
        "🚀 Project Analysis"
    )

    project_analysis = analysis.get(
        "project_analysis",
        ""
    )


    if project_analysis:

        st.write(
            project_analysis
        )

    else:

        st.write(
            "No project analysis available."
        )


    # ========================================================
    # EDUCATION ANALYSIS
    # ========================================================

    st.subheader(
        "🎓 Education Analysis"
    )

    education_analysis = analysis.get(
        "education_analysis",
        ""
    )


    if education_analysis:

        st.write(
            education_analysis
        )

    else:

        st.write(
            "No education analysis available."
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
    # RETRIEVED RESUME EVIDENCE
    # ========================================================

    st.subheader(
        "🔎 Retrieved Resume Evidence"
    )


    if retrieved_documents:

        for index, item in enumerate(
            retrieved_documents
        ):

            # similarity_search_with_score
            # returns:
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


            # ----------------------------------------------
            # Expander title
            # ----------------------------------------------

            if score is not None:

                title = (
                    f"Chunk {index + 1} "
                    f"| Distance: {score:.4f}"
                )

            else:

                title = (
                    f"Chunk {index + 1}"
                )


            # ----------------------------------------------
            # Display document
            # ----------------------------------------------

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


                # ------------------------------------------
                # Metadata
                # ------------------------------------------

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
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
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

    # ============================================================
    # FOOTER
    # ============================================================

    st.markdown(
        """
        <hr>

        <div style="
            text-align: center;
            padding: 20px 0 10px 0;
            color: #64748B;
            font-size: 14px;
        ">

            <div style="
                font-size: 16px;
                font-weight: 600;
                color: #334155;
                margin-bottom: 8px;
            ">
                🤖 AI Resume Matcher
            </div>

            <div>
                Analyze • Match • Improve
            </div>

            <div style="
                margin-top: 8px;
                font-size: 12px;
                color: #94A3B8;
            ">
                Built with Python • LangChain • Hugging Face • FAISS • GPT-OSS-120B • Streamlit
            </div>

            <div style="
                margin-top: 12px;
                font-size: 12px;
                color: #94A3B8;
            ">
                © 2026 AI Resume Matcher
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )