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
                            Ranked Candidates + Visualizations, Export (CSV/PDF)

New in this version:
  7. Bias-Reduction Mode -> optional PII redaction (name/email/phone/URL) before scoring
  8. Filter & Search      -> live filter of ranked candidates by name/score/recommendation
  9. Candidate Comparison -> side-by-side radar chart + table for selected candidates
  10. Skill-Gap Insights  -> pool-wide "most common missing/matched skills" charts
  11. Ask-the-Pool Q&A    -> free-text question answered by the LLM over all results
  12. Email Drafting      -> AI-drafted outreach/rejection/follow-up email per candidate
  13. PDF Export          -> full multi-candidate report as a downloadable PDF
  14. Session Save/Load   -> export results to JSON and reload later without re-running the pipeline
"""

import io
import os
import re
import json
import tempfile
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

try:
    # Newer LangChain versions ship the splitter in its own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older LangChain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import AuthenticationError, RateLimitError, APIError

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
st.sidebar.subheader("🛡️ Bias & Privacy")
anonymize_resumes = st.sidebar.checkbox(
    "Redact PII before scoring (Bias-Reduction Mode)",
    value=False,
    help=(
        "Strips emails, phone numbers, URLs, and mentions of the candidate's name from the "
        "resume text before it is sent to the LLM, to reduce the chance that personal "
        "identifiers influence the match score."
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Session")
loaded_session_file = st.sidebar.file_uploader(
    "Reload a previous session (JSON)", type=["json"], key="session_loader"
)
if loaded_session_file is not None:
    try:
        loaded_data = json.load(loaded_session_file)
        if st.sidebar.button("♻️ Restore this session"):
            st.session_state.results = loaded_data
            st.sidebar.success(f"Restored {len(loaded_data)} candidate(s).")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read session file: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption("Tech Stack: LangChain · OpenAI · FAISS · Streamlit · Plotly · FPDF2")

# Strip accidental whitespace/newlines from copy-pasted keys — a very common
# cause of a spurious AuthenticationError.
if api_key:
    api_key = api_key.strip()
    os.environ["OPENAI_API_KEY"] = api_key

if st.sidebar.button("🔑 Test API Key"):
    if not api_key:
        st.sidebar.error("Enter an API key above first.")
    else:
        try:
            OpenAIEmbeddings(model=embedding_model).embed_query("connection test")
            st.sidebar.success("✅ API key is valid and working.")
        except AuthenticationError:
            st.sidebar.error(
                "❌ Invalid API key. Check for typos, extra spaces, an expired/revoked key, "
                "or a key from a project without API access."
            )
        except RateLimitError:
            st.sidebar.error("⚠️ Key is valid, but you're rate-limited or out of quota/credits.")
        except APIError as exc:
            st.sidebar.error(f"⚠️ OpenAI API error: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"⚠️ Unexpected error: {exc}")

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


def anonymize_text(text: str, candidate_name: str) -> str:
    """Bias-Reduction Mode: redact emails, phone numbers, URLs, and the candidate's name
    from resume text before it is sent to the LLM for scoring."""
    text = re.sub(r"[\w.\-]+@[\w.\-]+\.\w+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"(\+?\d[\d\-.\s()]{7,}\d)", "[REDACTED_PHONE]", text)
    text = re.sub(r"https?://\S+", "[REDACTED_URL]", text)
    for part in re.split(r"[\s_\-]+", candidate_name):
        if len(part) > 2:
            text = re.sub(re.escape(part), "[CANDIDATE]", text, flags=re.IGNORECASE)
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
# EMAIL DRAFTING (AI-assisted candidate communication)
# --------------------------------------------------------------------------------------
EMAIL_PROMPT = """You are a professional recruiter writing on behalf of a hiring team.

Write a concise, warm, professional {tone} email to a job candidate named {candidate_name}.

Context:
- Match score: {score}%
- Hiring recommendation: {recommendation}
- Reason: {reason}

{extra_instruction}

Keep the whole email under 150 words. Output the subject line on the first line
(prefixed with "Subject:"), a blank line, then the email body. No other commentary.
"""


def draft_email(candidate_name: str, score: int, recommendation: str, reason: str) -> str:
    if recommendation == "Yes":
        tone = "interview invitation"
        extra = "Invite them to schedule a next-round interview and express genuine enthusiasm about their background."
    elif recommendation == "No":
        tone = "polite rejection"
        extra = "Thank them for their time, let them know the team is moving forward with other candidates, and wish them well."
    else:
        tone = "friendly follow-up"
        extra = "Ask one clarifying question related to a skill gap before a final decision is made."

    llm = ChatOpenAI(model=llm_model, temperature=0.4)
    prompt = EMAIL_PROMPT.format(
        candidate_name=candidate_name,
        score=score,
        recommendation=recommendation,
        reason=reason or "N/A",
        tone=tone,
        extra_instruction=extra,
    )
    response = llm.invoke(prompt)
    return response.content.strip()


# --------------------------------------------------------------------------------------
# PDF REPORT EXPORT
# --------------------------------------------------------------------------------------
def _pdf_safe(text) -> str:
    """Make LLM-generated text safe for fpdf2's core (Latin-1-only) fonts.

    LLM output frequently contains smart quotes, em-dashes, bullets, or other
    Unicode characters that the built-in Helvetica font can't render, which
    crashes fpdf2's line-wrapping. This normalizes common punctuation, strips
    anything else outside Latin-1, and breaks up pathologically long unbroken
    tokens (e.g. long URLs) so multi_cell always has somewhere to wrap.
    """
    text = "" if text is None else str(text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
        "\u2022": "-", "\u2192": "->",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Drop anything the core font still can't encode (emoji, CJK, etc.)
    text = text.encode("latin-1", "ignore").decode("latin-1")
    # Insert a break point into any unbroken run of 60+ characters
    text = re.sub(r"(\S{60})(?=\S)", r"\1 ", text)
    return text.strip() or " "


def build_pdf_report(report_df: pd.DataFrame) -> bytes:
    """Render a multi-candidate PDF report (one page per candidate)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def _section(title: str, items):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _pdf_safe(title), ln=True)
        pdf.set_font("Helvetica", "", 11)
        if items:
            for it in items:
                pdf.multi_cell(0, 6, f"- {_pdf_safe(it)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.multi_cell(0, 6, "None listed", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    for rank, row in report_df.iterrows():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _pdf_safe(f"#{rank} - {row['candidate_name']}"), ln=True)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(
            0, 8,
            _pdf_safe(f"Match Score: {row['match_score']}%   Recommendation: {row['recommendation']}"),
            ln=True,
        )
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Summary:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _pdf_safe(row.get("summary", "")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        _section("Strengths:", row.get("strengths", []))
        _section("Weaknesses:", row.get("weaknesses", []))
        _section("Matched Skills:", row.get("matched_skills", []))
        _section("Missing Skills:", row.get("missing_skills", []))
        _section("Suggested Interview Questions:", row.get("interview_questions", []))

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Recommendation Reason:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _pdf_safe(row.get("recommendation_reason", "")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    output = pdf.output(dest="S")
    return bytes(output)


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
            if anonymize_resumes:
                resume_text = anonymize_text(resume_text, candidate_name)

            try:
                result = analyze_candidate(jd_text, resume_text, candidate_name)
            except AuthenticationError:
                progress.empty()
                st.error(
                    "❌ OpenAI rejected your API key (AuthenticationError). This is a key/account "
                    "issue, not a pipeline bug. Check that the key in the sidebar has no extra "
                    "spaces, hasn't been revoked or expired, and belongs to a project with access "
                    "to the embeddings and chat APIs — use **🔑 Test API Key** in the sidebar to "
                    "confirm, then re-run."
                )
                st.stop()
            except RateLimitError:
                progress.empty()
                st.error(
                    "⚠️ Rate limit or quota exceeded on your OpenAI account. Check your usage/billing "
                    "at platform.openai.com and try again."
                )
                st.stop()
            except APIError as exc:
                progress.empty()
                st.error(f"⚠️ OpenAI API error while processing **{candidate_name}**: {exc}")
                st.stop()

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

    # ---- Filter & Search ----
    st.subheader("🔎 Filter & Search")
    fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
    with fcol1:
        search_term = st.text_input("Search by candidate name", "")
    with fcol2:
        reco_filter = st.multiselect(
            "Filter by recommendation", ["Yes", "Maybe", "No"], default=["Yes", "Maybe", "No"]
        )
    with fcol3:
        min_score = st.slider("Min. match score", 0, 100, 0)

    filtered_df = df[
        df["candidate_name"].str.contains(search_term, case=False, na=False)
        & df["recommendation"].isin(reco_filter)
        & (df["match_score"] >= min_score)
    ]
    st.caption(
        f"Showing {len(filtered_df)} of {len(df)} candidate(s). "
        "Filters affect the table, charts, and detail cards below — exports always include the full pool."
    )

    if filtered_df.empty:
        st.warning("No candidates match the current filters.")
    else:
        # ---- Ranked candidates table ----
        st.subheader("🏆 Ranked Candidates")
        display_df = filtered_df[["candidate_name", "match_score", "recommendation"]].copy()
        display_df["recommendation"] = display_df["recommendation"].map(lambda r: f"{reco_color.get(r,'')} {r}")
        display_df.columns = ["Candidate", "Match Score (%)", "Recommendation"]
        st.dataframe(display_df, use_container_width=True)

        # ---- Visualization 1: Bar chart comparing match scores ----
        st.subheader("📊 Match Score Comparison")
        fig_bar = px.bar(
            filtered_df,
            x="candidate_name",
            y="match_score",
            color="recommendation",
            color_discrete_map={"Yes": "#22c55e", "Maybe": "#eab308", "No": "#ef4444"},
            text="match_score",
            labels={
                "candidate_name": "Candidate",
                "match_score": "Match Score (%)",
                "recommendation": "Recommendation",
            },
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(yaxis_range=[0, 110])
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---- Candidate Comparison ----
        st.subheader("⚖️ Compare Candidates")
        compare_names = st.multiselect(
            "Select 2 or more candidates to compare side-by-side",
            options=filtered_df["candidate_name"].tolist(),
        )
        if len(compare_names) >= 2:
            compare_df = filtered_df[filtered_df["candidate_name"].isin(compare_names)]

            max_matched = max(compare_df["matched_skills"].apply(len).max(), 1)
            max_strengths = max(compare_df["strengths"].apply(len).max(), 1)
            max_missing = max(compare_df["missing_skills"].apply(len).max(), 1)
            categories = ["Match Score", "Matched Skills", "Strengths", "Few Skill Gaps"]

            fig_radar = go.Figure()
            for _, crow in compare_df.iterrows():
                values = [
                    crow["match_score"],
                    (len(crow["matched_skills"]) / max_matched) * 100,
                    (len(crow["strengths"]) / max_strengths) * 100,
                    (1 - len(crow["missing_skills"]) / max_missing) * 100,
                ]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=crow["candidate_name"],
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            compare_table = compare_df[["candidate_name", "match_score", "recommendation"]].copy()
            compare_table.columns = ["Candidate", "Match Score (%)", "Recommendation"]
            st.dataframe(compare_table, use_container_width=True)
        elif len(compare_names) == 1:
            st.info("Select at least one more candidate to compare.")

        # ---- Skill-Gap Insights across the pool ----
        st.subheader("📉 Skill Gaps Across Candidate Pool")
        all_missing = Counter()
        all_matched = Counter()
        for _, r in filtered_df.iterrows():
            all_missing.update([s.strip() for s in r.get("missing_skills", []) if s.strip()])
            all_matched.update([s.strip() for s in r.get("matched_skills", []) if s.strip()])

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            if all_missing:
                top_missing = pd.DataFrame(all_missing.most_common(10), columns=["Skill", "Candidates Missing"])
                fig_missing = px.bar(
                    top_missing, x="Candidates Missing", y="Skill", orientation="h",
                    color_discrete_sequence=["#ef4444"], title="Most Common Missing Skills",
                )
                fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.caption("No missing-skill data available for the current filter.")
        with gcol2:
            if all_matched:
                top_matched = pd.DataFrame(all_matched.most_common(10), columns=["Skill", "Candidates Matched"])
                fig_matched = px.bar(
                    top_matched, x="Candidates Matched", y="Skill", orientation="h",
                    color_discrete_sequence=["#22c55e"], title="Most Common Matched Skills",
                )
                fig_matched.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_matched, use_container_width=True)
            else:
                st.caption("No matched-skill data available for the current filter.")

        # ---- Per-candidate detail cards ----
        st.subheader("🔍 Candidate Details")
        for rank, row in filtered_df.iterrows():
            with st.expander(
                f"#{rank} — {row['candidate_name']}  ({reco_color.get(row['recommendation'],'')} {row['recommendation']})"
            ):
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

                # ---- AI Email Drafting ----
                st.markdown("---")
                st.markdown("**✉️ Candidate Communication**")
                email_key = f"email_{rank}_{row['candidate_name']}"
                if st.button(f"Draft {row['recommendation']}-track email", key=f"btn_{email_key}"):
                    if not api_key:
                        st.error("Please enter your OpenAI API key in the sidebar first.")
                    else:
                        try:
                            with st.spinner("Drafting email..."):
                                st.session_state[email_key] = draft_email(
                                    row["candidate_name"],
                                    row["match_score"],
                                    row["recommendation"],
                                    row.get("recommendation_reason", ""),
                                )
                        except AuthenticationError:
                            st.error("❌ Invalid OpenAI API key. Use 🔑 Test API Key in the sidebar to confirm.")
                        except RateLimitError:
                            st.error("⚠️ Rate limit or quota exceeded on your OpenAI account.")
                        except APIError as exc:
                            st.error(f"⚠️ OpenAI API error: {exc}")
                if email_key in st.session_state:
                    st.text_area(
                        "Draft email (editable before sending)",
                        value=st.session_state[email_key],
                        height=220,
                        key=f"ta_{email_key}",
                    )

    # ---- Ask-the-Pool Q&A ----
    st.subheader("💬 Ask About This Candidate Pool")
    user_question = st.text_input(
        "e.g. \"Which candidates have both Python and AWS experience?\" or \"Who has the strongest leadership background?\""
    )
    if st.button("Ask") and user_question:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar first.")
        else:
            try:
                with st.spinner("Thinking..."):
                    pool_context = json.dumps(
                        [
                            {
                                "name": r["candidate_name"],
                                "score": r["match_score"],
                                "recommendation": r["recommendation"],
                                "matched_skills": r.get("matched_skills", []),
                                "missing_skills": r.get("missing_skills", []),
                                "summary": r.get("summary", ""),
                            }
                            for r in results
                        ],
                        indent=2,
                    )
                    qa_prompt = f"""You are a recruiting assistant. Here is structured data about candidates
who were screened against a job description:

{pool_context}

Answer the recruiter's question using ONLY this data. Be concise, reference candidate
names directly, and say so plainly if the data doesn't contain the answer.

Question: {user_question}
"""
                    qa_llm = ChatOpenAI(model=llm_model, temperature=0.2)
                    answer = qa_llm.invoke(qa_prompt).content.strip()
                st.markdown(f"**Answer:** {answer}")
            except AuthenticationError:
                st.error("❌ Invalid OpenAI API key. Use 🔑 Test API Key in the sidebar to confirm.")
            except RateLimitError:
                st.error("⚠️ Rate limit or quota exceeded on your OpenAI account.")
            except APIError as exc:
                st.error(f"⚠️ OpenAI API error: {exc}")

    # ---- Export ----
    st.subheader("📤 Export Reports")
    st.caption("Exports always include the full candidate pool, regardless of the filters above.")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv_buf.getvalue(),
            file_name="candidate_match_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        try:
            pdf_bytes = build_pdf_report(df)
            st.download_button(
                "⬇️ Download Full Report as PDF",
                data=pdf_bytes,
                file_name="candidate_match_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"⚠️ Couldn't generate the PDF report ({exc}). CSV and JSON exports are unaffected.")

    with exp_col3:
        st.download_button(
            "💾 Save Session (JSON)",
            data=json.dumps(results, indent=2),
            file_name="rag_session.json",
            mime="application/json",
            use_container_width=True,
            help="Reload this file from the sidebar later to revisit results without re-running the pipeline.",
        )

    st.markdown("##### 👀 Report Preview")
    st.caption("This is a visual preview of exactly what's inside the CSV/PDF above — one card per candidate, ranked.")

    reco_style = {
        "Yes":   {"bg": "#dcfce7", "border": "#22c55e", "text": "#15803d", "emoji": "🟢"},
        "Maybe": {"bg": "#fef9c3", "border": "#eab308", "text": "#a16207", "emoji": "🟡"},
        "No":    {"bg": "#fee2e2", "border": "#ef4444", "text": "#b91c1c", "emoji": "🔴"},
    }

    def _pills(items, color):
        if not items:
            return "<span style='color:#9ca3af; font-size:13px;'>None listed</span>"
        return "".join(
            f"<span style='display:inline-block; background:{color}; border-radius:999px; "
            f"padding:3px 10px; margin:2px; font-size:12px; color:#111827;'>{i}</span>"
            for i in items
        )

    for rank, row in df.iterrows():
        reco = row.get("recommendation", "Maybe")
        style = reco_style.get(reco, reco_style["Maybe"])
        score = row.get("match_score", 0)

        card_html = f"""
        <div style="border:1px solid {style['border']}; border-left:6px solid {style['border']};
                    border-radius:12px; padding:16px 20px; margin-bottom:14px; background:#ffffff;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
              <span style="font-size:12px; color:#6b7280;">RANK #{rank}</span><br/>
              <span style="font-size:18px; font-weight:700; color:#111827;">{row['candidate_name']}</span>
            </div>
            <div style="text-align:right;">
              <div style="font-size:26px; font-weight:800; color:#111827;">{score}%</div>
              <div style="background:{style['bg']}; color:{style['text']}; border-radius:999px;
                          padding:4px 14px; font-size:13px; font-weight:600; display:inline-block; margin-top:4px;">
                {style['emoji']} {reco}
              </div>
            </div>
          </div>

          <div style="width:100%; background:#f3f4f6; border-radius:999px; height:8px; margin:12px 0;">
            <div style="width:{score}%; background:{style['border']}; height:8px; border-radius:999px;"></div>
          </div>

          <p style="font-size:14px; color:#374151; margin:8px 0 12px 0;"><em>{row.get('summary','')}</em></p>

          <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div>
              <div style="font-size:12px; font-weight:700; color:#15803d; margin-bottom:4px;">✅ STRENGTHS</div>
              {_pills(row.get('strengths', []), '#dcfce7')}
            </div>
            <div>
              <div style="font-size:12px; font-weight:700; color:#b91c1c; margin-bottom:4px;">⚠️ WEAKNESSES</div>
              {_pills(row.get('weaknesses', []), '#fee2e2')}
            </div>
            <div>
              <div style="font-size:12px; font-weight:700; color:#1d4ed8; margin-bottom:4px;">🎯 MATCHED SKILLS</div>
              {_pills(row.get('matched_skills', []), '#dbeafe')}
            </div>
            <div>
              <div style="font-size:12px; font-weight:700; color:#a16207; margin-bottom:4px;">❌ MISSING SKILLS</div>
              {_pills(row.get('missing_skills', []), '#fef9c3')}
            </div>
          </div>

          <div style="margin-top:12px; padding-top:10px; border-top:1px dashed #e5e7eb;">
            <span style="font-size:12px; font-weight:700; color:#6b7280;">💬 WHY: </span>
            <span style="font-size:13px; color:#374151;">{row.get('recommendation_reason','')}</span>
          </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
else:
    st.info("👆 Upload a Job Description and one or more resumes, then click **Run RAG Pipeline** to see results.")
