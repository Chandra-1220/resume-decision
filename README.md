# GenAI-Powered Resume Screening & Candidate Matching Platform

**An intelligent recruitment platform that automates resume screening, candidate ranking, and hiring recommendations using Retrieval-Augmented Generation (RAG).**

Built with OpenAI Embeddings, FAISS Vector Search, and GPT-4o / GPT-4o-mini, the platform helps recruiters move beyond keyword matching to true semantic understanding of resumes against job requirements.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Generated Insights](#generated-insights)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)


---

## Overview

Traditional resume screening is time-consuming and often relies on rigid keyword matching, which can overlook qualified candidates whose resumes are phrased differently from the job description.

This project applies **Generative AI** and **Retrieval-Augmented Generation (RAG)** to perform intelligent, semantic matching between resumes and job descriptions — surfacing the most relevant candidates based on meaning, not just vocabulary.

With this platform, recruiters can:

- Upload multiple resumes at once
- Upload a job description (JD)
- Run semantic candidate matching against the JD
- Automatically rank candidates by fit
- Generate AI-powered hiring recommendations
- Identify missing or gap skills per candidate
- Ask natural-language questions across the entire candidate pool
- Export results as CSV or PDF reports

---

## System Architecture

> End-to-end system architecture for a GenAI-powered resume screening and candidate matching platform using Retrieval-Augmented Generation (RAG).

<p align="center">
  <img src="docs/architecture.png" width="100%" alt="System architecture diagram">
</p>

---

## Features

### 📄 Resume Processing
- Bulk resume upload
- Supported formats: **PDF, DOCX, TXT**

### 📋 Job Description Processing
- JD upload and parsing
- Semantic understanding of role requirements
- Automatic requirement extraction

### 🎯 Intelligent Candidate Matching
- Resume and JD embedding generation
- Semantic similarity search via FAISS
- Automated candidate ranking
- Match score generation

### 🧠 AI Candidate Evaluation
- Concise candidate summaries
- Strengths and weaknesses breakdown
- Skill-gap analysis
- Hiring recommendation
- Suggested interview questions

### ⚖️ Candidate Comparison
- Side-by-side comparison of multiple candidates
- Skill-by-skill comparison
- Match score comparison

### 📊 Analytics Dashboard
- Candidate ranking view
- Match score visualization
- Skill distribution charts
- Missing-skills analysis

### 🤝 AI Recruitment Assistant
- Ask questions about the entire candidate pool
- AI-generated outreach and follow-up emails
- Candidate shortlisting recommendations

### 📤 Export Options
- CSV reports
- PDF reports
- JSON session export

---

## Technology Stack

| Category          | Technology                     |
|--------------------|---------------------------------|
| Frontend           | Streamlit                      |
| Language            | Python                          |
| LLM                 | GPT-4o / GPT-4o-mini            |
| Framework           | LangChain                       |
| Embeddings          | OpenAI `text-embedding-3-small` |
| Vector Database     | FAISS                           |
| Data Processing     | Pandas                          |
| Visualization       | Plotly                          |
| PDF Report Generation | FPDF2                         |
| Storage             | JSON                            |
| Supported File Formats | PDF, DOCX, TXT               |

---

## Project Structure

```text
Resume-Screening-RAG/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── docs/
│   └── architecture.png       # System architecture diagram
│
├── data/
│   ├── resumes/                # Uploaded resume files
│   └── job_descriptions/       # Uploaded job descriptions
│
├── embeddings/                 # Generated embeddings
├── vector_store/                # FAISS vector index
├── reports/                     # Exported CSV/PDF reports
├── utils/                       # Helper utilities
├── services/                    # Core business logic (matching, evaluation, etc.)
└── assets/                      # Static assets
```

---

## Workflow

```text
Recruiter
   │
   ▼
Upload Job Description ──▶ Upload Resumes
   │
   ▼
Document Processing
   │
   ├─ Text Extraction
   ├─ Chunking
   └─ Metadata Generation
   │
   ▼
OpenAI Embeddings
   │
   ▼
FAISS Vector Database
   │
   ▼
Similarity Search ──▶ Top-K Resume Chunks
   │
   ▼
RAG Prompt Construction
   │
   ▼
GPT-4o Evaluation
   │
   ▼
Ranking ──▶ Visualization ──▶ Reports ──▶ Email Drafts ──▶ Exports
```

---

## Generated Insights

The platform automatically generates, for every candidate:

- ✅ Match score
- ✅ Candidate summary
- ✅ Strengths
- ✅ Weaknesses
- ✅ Skill-gap analysis
- ✅ Missing skills
- ✅ Suggested interview questions
- ✅ Hiring recommendation
- ✅ Candidate ranking
- ✅ Analytics dashboard view

### Sample Output

After analyzing uploaded resumes against a job description, the platform provides:

- Candidate match score and ranking
- Strengths and weaknesses per candidate
- Skill-gap analysis
- AI-generated hiring recommendation
- Suggested interview questions
- Candidate comparison view
- Match score distribution chart
- Exportable CSV/PDF reports
- AI-generated recruitment emails

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/Resume-Screening-RAG.git
```

**2. Navigate to the project directory**

```bash
cd Resume-Screening-RAG
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your OpenAI API key**

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key
```

**5. Run the application**

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` by default.

---

## Usage

1. Launch the app with `streamlit run app.py`.
2. Upload a job description (PDF, DOCX, or TXT).
3. Upload one or more candidate resumes.
4. Let the platform generate embeddings and run semantic matching.
5. Review ranked candidates, match scores, and AI evaluations.
6. Compare candidates side by side, if needed.
7. Export results as CSV, PDF, or JSON.
8. Use the AI recruitment assistant to draft outreach emails or ask questions about the candidate pool.

---

## Future Enhancements

- [ ] Multi-LLM support
- [ ] Resume OCR for scanned documents
- [ ] LinkedIn profile analysis
- [ ] ATS integration
- [ ] PostgreSQL support
- [ ] ChromaDB support
- [ ] Pinecone integration
- [ ] Azure OpenAI support
- [ ] Multi-language resume analysis
- [ ] Authentication & role-based access control

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

