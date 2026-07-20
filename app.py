"""
Resume Screening & Candidate Matching — GenAI RAG Architecture
================================================================
Pipeline (matches the reference architecture diagram):
  1. Data Ingestion      -> upload resumes (PDF/DOCX/TXT) + Job Description
  2. Document Processing -> text extraction + chunking + metadata
  3. Embeddings & Vector Store -> OpenAI Embeddings + FAISS (per-candidate index)
  4. Retrieval           -> embed JD query, similarity search top-k chunks
  5. LLM Generation (RAG)-> prompt + retrieved context -> structured JSON output
  6. Output              -> Match Score, Strengths, Weaknesses, Skill Gaps,
                            Summary, Interview Questions, Hiring Recommendation,
                            Ranked Candidates + Visualizations, Export (CSV)
"""

import io
import os
import json
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

try:
    # Newer LangChain versions ship the splitter in its own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older LangChain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Screening & Candidate Matching (RAG)",
    page_icon="🧠",
    layout="wide",
)

if "results" not in st.session_state:
    st.session_state.results = []

# --------------------------------------------------------------------------------------
# SIDEBAR — CONFIG (STEP 0: credentials & model)
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Your key is used only for this session and never stored.",
)
llm_model = st.sidebar.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=1)
embedding_model = st.sidebar.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])
top_k = st.sidebar.slider("Top-K Retrieved Chunks", 2, 10, 5)
chunk_size = st.sidebar.slider("Chunk Size", 300, 1500, 800, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 100, step=50)

st.sidebar.markdown("---")
st.sidebar.caption("Tech Stack: LangChain · OpenAI · FAISS · Streamlit")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.title("🧠 Resume Screening & Candidate Matching")
st.caption("Retrieve relevant resume information and generate hiring insights using an LLM (RAG architecture)")

# --------------------------------------------------------------------------------------
# STEP 1: DATA INGESTION
# --------------------------------------------------------------------------------------
st.header("① Data Ingestion")
col1, col2 = st.columns(2)

with col1:
    jd_file = st.file_uploader("📄 Upload Job Description (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    jd_text_manual = st.text_area("...or paste the Job Description here", height=150)

with col2:
    resume_files = st.file_uploader(
        "📎 Upload Candidate Resumes (bulk upload supported)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

process_btn = st.button("🚀 Run RAG Pipeline (Process & Match)", type="primary", use_container_width=True)


# --------------------------------------------------------------------------------------
# STEP 2: DOCUMENT PROCESSING  (loader -> extraction -> chunking -> metadata)
# --------------------------------------------------------------------------------------
def load_document_text(uploaded_file) -> str:
    """Document Loader + Text Extraction for PDF / DOCX / TXT."""
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs)
    finally:
        os.remove(tmp_path)
    return text


def chunk_text(text: str, candidate_name: str):
    """Text Chunking (RecursiveCharacterTextSplitter) + Metadata Enrichment."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents(
        [text], metadatas=[{"candidate": candidate_name}]
    )
    return chunks


# --------------------------------------------------------------------------------------
# STEP 3 & 4: EMBEDDINGS/VECTOR STORE + RETRIEVAL
# --------------------------------------------------------------------------------------
def build_vectorstore(chunks):
    """Generate Embeddings + Vector Database (FAISS)."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.from_documents(chunks, embeddings)


def retrieve_top_chunks(vectorstore, query: str, k: int):
    """Embed query -> similarity search -> top-K relevant chunks."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


# --------------------------------------------------------------------------------------
# STEP 5: LLM GENERATION (RAG) — structured JSON output
# --------------------------------------------------------------------------------------
ANALYSIS_PROMPT = """You are an expert technical recruiter AI performing candidate screening.

JOB DESCRIPTION:
{jd_text}

RETRIEVED RESUME CONTEXT (most relevant chunks for candidate "{candidate_name}"):
{context}

Based ONLY on the information above, evaluate this candidate against the job description.
Respond with STRICT JSON ONLY (no markdown fences, no preamble) matching this exact schema:

{{
  "candidate_name": "{candidate_name}",
  "match_score": <integer 0-100>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "missing_skills": ["...", "..."],
  "matched_skills": ["...", "..."],
  "summary": "2-3 sentence resume summary",
  "interview_questions": ["...", "...", "..."],
  "recommendation": "Yes" | "Maybe" | "No",
  "recommendation_reason": "1-2 sentence detailed justification for the hiring recommendation"
}}
"""


def analyze_candidate(jd_text: str, resume_text: str, candidate_name: str) -> dict:
    chunks = chunk_text(resume_text, candidate_name)
    vectorstore = build_vectorstore(chunks)
    context = retrieve_top_chunks(vectorstore, jd_text, k=top_k)

    llm = ChatOpenAI(model=llm_model, temperature=0.2)
    prompt = ANALYSIS_PROMPT.format(jd_text=jd_text, context=context, candidate_name=candidate_name)
    response = llm.invoke(prompt)

    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "candidate_name": candidate_name,
            "match_score": 0,
            "strengths": [],
            "weaknesses": ["Could not parse model output"],
            "missing_skills": [],
            "matched_skills": [],
            "summary": raw[:300],
            "interview_questions": [],
            "recommendation": "Maybe",
            "recommendation_reason": "Automatic parsing failed; review manually.",
        }
    return data


# --------------------------------------------------------------------------------------
# PIPELINE EXECUTION
# --------------------------------------------------------------------------------------
if process_btn:
    jd_text = jd_text_manual.strip()
    if jd_file is not None:
        jd_text = load_document_text(jd_file)

    if not jd_text:
        st.error("Please provide a Job Description (upload a file or paste text).")
    elif not resume_files:
        st.error("Please upload at least one candidate resume.")
    elif not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        results = []
        progress = st.progress(0.0, text="Starting RAG pipeline...")
        for i, rf in enumerate(resume_files):
            candidate_name = rf.name.rsplit(".", 1)[0]
            progress.progress((i) / len(resume_files), text=f"Processing {candidate_name}...")
            resume_text = load_document_text(rf)
            result = analyze_candidate(jd_text, resume_text, candidate_name)
            results.append(result)
            progress.progress((i + 1) / len(resume_files), text=f"Done: {candidate_name}")
        progress.empty()
        st.session_state.results = results
        st.success(f"✅ Processed {len(results)} candidate(s) through the RAG pipeline.")


# --------------------------------------------------------------------------------------
# STEP 6: OUTPUT — Ranked candidates, scores, visualizations, details
# --------------------------------------------------------------------------------------
results = st.session_state.results

if results:
    st.header("⑥ Output — Results")

    df = pd.DataFrame(results).sort_values("match_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank starting at 1

    reco_color = {"Yes": "🟢", "Maybe": "🟡", "No": "🔴"}

    # ---- Ranked candidates table ----
    st.subheader("🏆 Ranked Candidates")
    display_df = df[["candidate_name", "match_score", "recommendation"]].copy()
    display_df["recommendation"] = display_df["recommendation"].map(lambda r: f"{reco_color.get(r,'')} {r}")
    display_df.columns = ["Candidate", "Match Score (%)", "Recommendation"]
    st.dataframe(display_df, use_container_width=True)

    # ---- Visualization 1: Bar chart comparing match scores ----
    st.subheader("📊 Match Score Comparison")
    fig_bar = px.bar(
        df,
        x="candidate_name",
        y="match_score",
        color="recommendation",
        color_discrete_map={"Yes": "#22c55e", "Maybe": "#eab308", "No": "#ef4444"},
        text="match_score",
        labels={"candidate_name": "Candidate", "match_score": "Match Score (%)", "recommendation": "Recommendation"},
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_layout(yaxis_range=[0, 110])
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Per-candidate detail cards ----
    st.subheader("🔍 Candidate Details")
    for rank, row in df.iterrows():
        with st.expander(f"#{rank} — {row['candidate_name']}  ({reco_color.get(row['recommendation'],'')} {row['recommendation']})"):
            c1, c2 = st.columns([1, 2])

            with c1:
                # Gauge visualization for individual match score
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=row["match_score"],
                    title={"text": "Match Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563eb"},
                        "steps": [
                            {"range": [0, 50], "color": "#fee2e2"},
                            {"range": [50, 75], "color": "#fef9c3"},
                            {"range": [75, 100], "color": "#dcfce7"},
                        ],
                    },
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown(f"**Recommendation:** {reco_color.get(row['recommendation'],'')} **{row['recommendation']}**")
                st.caption(row.get("recommendation_reason", ""))

            with c2:
                st.markdown("**📝 Resume Summary**")
                st.write(row.get("summary", ""))

                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**✅ Strengths**")
                    for s in row.get("strengths", []):
                        st.markdown(f"- {s}")
                    st.markdown("**🎯 Matched Skills**")
                    for s in row.get("matched_skills", []):
                        st.markdown(f"- {s}")
                with colB:
                    st.markdown("**⚠️ Weaknesses**")
                    for w in row.get("weaknesses", []):
                        st.markdown(f"- {w}")
                    st.markdown("**❌ Missing Skills (Gap)**")
                    for m in row.get("missing_skills", []):
                        st.markdown(f"- {m}")

                st.markdown("**❓ Suggested Interview Questions**")
                for q in row.get("interview_questions", []):
                    st.markdown(f"- {q}")

    # ---- Export ----
    st.subheader("📤 Export Reports")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Results as CSV",
        data=csv_buf.getvalue(),
        file_name="candidate_match_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("👆 Upload a Job Description and one or more resumes, then click **Run RAG Pipeline** to see results.")
