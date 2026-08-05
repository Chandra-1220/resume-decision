# Resume Screening & Candidate Matching (RAG)

An HR tool that screens candidate resumes against a job description using a
Retrieval-Augmented Generation (RAG) pipeline, and gives recruiters ranked
results, comparisons, skill-gap insights, AI-drafted emails, and exportable
reports — all from a single Streamlit app.

## Features

- **Bulk resume screening** — upload a job description and any number of
  resumes (PDF/DOCX/TXT), and get a match score, strengths, weaknesses,
  missing/matched skills, a summary, interview questions, and a hiring
  recommendation for each candidate.
- **RAG pipeline** — resumes are chunked, embedded, and stored in a
  per-candidate FAISS index; the job description is used as the retrieval
  query so only the most relevant resume chunks are sent to the LLM.
- **Bias-reduction mode** — optionally redact emails, phone numbers, URLs,
  and the candidate's name from resume text before it's scored.
- **Filter & search** — live filtering of ranked candidates by name,
  recommendation, and minimum match score. Filters apply to the table,
  charts, candidate detail cards, and the report preview.
- **Candidate comparison** — side-by-side radar chart and table for any
  candidates you select.
- **Skill-gap insights** — pool-wide charts of the most common missing and
  matched skills across all candidates.
- **Ask-the-pool Q&A** — ask a free-text question (e.g. "who has both Python
  and AWS?") and get an answer grounded in the screened candidate data.
- **Email drafting** — AI-drafted interview invitation, rejection, or
  follow-up email per candidate, editable before sending.
- **Exports** — download results as CSV, a full multi-candidate PDF report,
  or a JSON session file you can reload later without re-running the
  pipeline. Exports always include the full candidate pool, independent of
  any active filters.

## Architecture

```
Data ingestion (JD + resumes)
        |
Document processing (extract, chunk, redact PII)
        |
Embeddings + FAISS (per-candidate vector index)
        |
Retrieval (top-k similarity search against the JD)
        |
LLM generation (structured JSON match analysis)
        |
Output (ranked table, charts, detail cards, exports)
```

Built with LangChain, OpenAI (embeddings + chat), FAISS, Streamlit, Plotly,
and FPDF2.

## Setup

1. Install dependencies:
   ```bash
   pip install streamlit langchain langchain-community langchain-openai \
               langchain-text-splitters faiss-cpu openai pandas plotly fpdf2 \
               pypdf docx2txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Enter your OpenAI API key in the sidebar. It's kept only in your own
   browser session (`st.session_state`) — see **Security** below.

## Usage

1. Upload or paste a job description.
2. Upload one or more candidate resumes.
3. Click **Run RAG Pipeline** to process and score every candidate.
4. Use the filter controls to narrow the results, compare candidates,
   review skill gaps, ask questions about the pool, draft outreach emails,
   and export a report.
5. Use **Save Session (JSON)** to export your results, and the sidebar's
   **Reload a previous session** uploader to restore them later without
   re-running the pipeline.

## Security notes

- The API key lives only in `st.session_state`, which Streamlit scopes to
  each individual browser session — it is never written to a process-wide
  variable like `os.environ`. That means one visitor's key can never be
  pre-filled into, or reused by, another visitor's session, even when
  multiple people use the same deployed instance concurrently.
- Every OpenAI client call (`OpenAIEmbeddings`, `ChatOpenAI`) is passed the
  key explicitly via `api_key=`, so nothing can silently fall back to a
  stray global or environment value.
- **Bias-reduction mode** strips PII from resume text before it's sent to
  the LLM for scoring, but it does not change what's stored in your own
  session state, exports, or the on-screen candidate details.

## Notes & limitations

- Match scores, summaries, and recommendations are LLM-generated and should
  be used as a screening aid, not a sole basis for hiring decisions.
- The FAISS index is rebuilt per candidate per run and is not persisted —
  only the structured results (scores, summaries, etc.) are saved via the
  session JSON export.
- PDF export uses Latin-1-safe text rendering; unusual Unicode characters
  in LLM output are normalized or stripped to avoid render failures.
