# RTGS Analyst — Data-Agnostic Evidence Pipeline & Semantic Q&A

---

## 📌 Overview

RTGS Analyst is a **data-agnostic** pipeline that ingests structured CSVs, auto-detects & cleans key columns (dates, metrics, locations), validates data, builds a RAG (Retrieval-Augmented Generation) index, and provides **semantic Q&A** both via **CLI** and a **Streamlit dashboard**.

The pipeline is designed to adapt automatically — as long as your dataset has **dates / categorical fields / numeric metrics**, the system will attempt to infer the right mappings.

---

## 🚀 Live Demo Links

-   **Streamlit App:** [Streamlit Deployment URL](https://rtgs-agent.streamlit.app/)
-   **Demo Video (≤ 5 min):** [Google Drive Demo Link](https://drive.google.com/file/d/1OK2U1g-jNSpA9CVpIF5BUIWPRES0E0y5/view?usp=sharing)

---

## 🛠 Quick Start

### 1. Run the whole pipeline

python src/langgraph_shim.py --config configs/config.yaml --auto-approve

- This runs the full pipeline: ingest → clean → validate → RAG build → QA.

### 2. Interactive Semantic CLI

python src/semantic_cli.py --rag_dir outputs/rag --k 6 --model gemini-2.5-pro --temp 0.0

- Type questions interactively. Example:

  Question> Which sections have the highest total number of bookings in the dataset?
  
 After you are done with the questions click "quit"

### 3. Quick Analytics

python src/quick_analytics.py --in data/cleaned/tankers_cleaned_enhanced.csv \ --groupby section --metric noofbookings --top 10

This prints top-k sections and saves a CSV + PNG in outputs/plots.

### 4. Run the Streamlit Dashboard

streamlit run src/app_streamlit.py

--- 

## ⚙️ Requirements & Setup

### Install Dependencies
pip install -r requirements.txt

### Gemini API Keys

### For Streamlit Cloud → secrets.toml
[secrets]
GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxx"

GEMINI_MODEL = "gemini-2.5-pro"

### Local .env file
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxx

GEMINI_MODEL=gemini-2.5-pro

DEFAULT_TEMPERATURE=0.0

DEFAULT_MAX_TOKENS=300
---

## 🔑 Why This Is Data-Agnostic

- Column Detection Heuristics — auto-detects likely date, month, noofbookings, delivered, section, etc.
- Filename/Source Inference — extracts year/month from _source_file (e.g., tankers_reports_2024_3.csv).
- Flexible RAG Builder — generates passages regardless of dataset schema.
- Fallback Analytics — if LLM says "Insufficient evidence", deterministic numeric summaries run automatically.

---

## 🖥 Demo Flow
### 1. Run the full pipeline:
python src/langgraph_shim.py --config configs/config.yaml --auto-approve

### 2. Ask 1–2 questions in CLI:
python src/semantic_cli.py --rag_dir outputs/rag --k 6 --model gemini-2.5-pro --temp 0.0

### 3. Run quick analytics:
python src/quick_analytics.py --in data/cleaned/tankers_cleaned_enhanced.csv --groupby section --metric noofbookings --top 10

### 4. Launch Streamlit dashboard:
streamlit run src/app_streamlit.py

---

## 📂 Project Outputs

All generated outputs are organized under the `outputs/` folder with clear subdirectories:

- `outputs/plots/` → Charts & summaries (e.g., top sections by number of bookings).
- `outputs/profiles/` → Data profiles & schema mappings for auditability.
- `outputs/final/` → Curated final outputs for submission:
  - Evidence JSON
  - Validation reports
  - Profiles
  - Final report markdown
- `outputs/rag/` → Passages and embeddings used for semantic search.

This ensures the pipeline remains **data-agnostic** and can adapt to any structured CSV input with minimal changes.

---

## 🏗️ System Architecture

Below is the high-level flow of the system:

1. **Raw Data Ingestion** → Merge multiple tanker report CSVs into a unified dataset.  
2. **Enhanced Cleaning & Mapping** → Canonicalize column names (`noofbookings`, `delivered`, `date`, etc.), infer missing values, and verify mappings.  
3. **Validation & Profiling** → Generate schema mappings, quality reports, and profiles.  
4. **RAG Builder** → Build passages + embeddings for semantic search.  
5. **Semantic Q&A** → Evidence-grounded question answering using Gemini + deterministic fallback.  
6. **Analytics** → Quick summaries and charts for policy insights.  
7. **Interfaces** →  
   - **CLI** for analysts (terminal interaction).  
   - **Streamlit Dashboard** for interactive exploration.  

--- 

## Highlights
- Data-Agnostic Design — no hardcoding to tanker dataset, adaptable to any structured CSV.
- Evidence-Focused Q&A — avoids hallucination; LLM says "Insufficient evidence" when data is missing.
- Fallback Analytics — ensures numeric summaries are always available.
- Dual UX — CLI for reproducibility, Streamlit for user-friendly dashboard.
- End-to-End Pipeline — ingestion, cleaning, validation, RAG, Q&A in one command.

---

**Project Owner:** Shanmukha Mahathi Chinta
