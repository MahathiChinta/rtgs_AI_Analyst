
"""
Streamlit dashboard for the RTGS agent (clean, simple, polished).

Features:
 - Choose dataset (defaults to data/cleaned/tankers_cleaned_enhanced.csv)
 - Quick summary / DQ counts
 - Top-K by metric (bookings / delivered) + horizontal bar chart
 - Pre-canned QA questions (2 safe ones) + custom question input
 - Run semantic QA (calls src/semantic_qa.py) and shows report/evidence
 - Option to run full pipeline orchestrator (langgraph_shim.py) for one-click demo
"""
from pathlib import Path
import subprocess
import shlex
import json
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "cleaned" / "tankers_cleaned_enhanced.csv"
OUT_DIR = ROOT / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RTGS Agent — Demo", layout="wide", initial_sidebar_state="expanded")

# ---------- Sidebar ----------
st.sidebar.header("Settings & Actions")
data_choice = st.sidebar.file_uploader("Upload CSV (optional) — or use default dataset", type=["csv"])
use_default = False
if data_choice is None:
    if DEFAULT_CSV.exists():
        use_default = True
        data_path = DEFAULT_CSV
    else:
        st.sidebar.error("No default dataset found; please upload a CSV.")
        data_path = None
else:
    # save uploaded to a temp file
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tf.write(data_choice.getvalue())
    tf.flush()
    data_path = Path(tf.name)

# Quick orchestrator button (optional: runs entire pipeline)
run_pipeline = st.sidebar.button("Run full pipeline (ingest → clean → rag → qa)")

st.sidebar.markdown("---")
st.sidebar.write("Preloaded demo questions")
demo_q1 = "Which sections have the highest total number of bookings in the dataset? List the top 10 with counts."
demo_q2 = "List the top 10 sections by total delivered tankers. Provide the counts."
st.sidebar.markdown(f"- Q1: {demo_q1}")
st.sidebar.markdown(f"- Q2: {demo_q2}")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Run analytics first, then ask the QA question to generate evidence + report.")

# ---------- Main UI ----------
st.title("RTGS AI Agent — Demo Dashboard")
st.write("Clean, data-agnostic dashboard for quick analytics and semantic QA on tabular datasets.")

if data_path is None:
    st.stop()

# Load dataset (defensive)
@st.cache_data(max_entries=2)
def load_df(path):
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")
    # normalize empty strings -> NaN for counts
    df = df.replace({"": pd.NA})
    return df

try:
    df = load_df(data_path)
except Exception as e:
    st.error(str(e))
    st.stop()

# Summary cards
col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
# detect numeric booking/delivered candidates
booking_col = next((c for c in df.columns if "book" in c.lower()), None)
delivered_col = next((c for c in df.columns if "deliv" in c.lower()), None)
col3.metric("Bookings column", booking_col or "—")
col4.metric("Delivered column", delivered_col or "—")

st.markdown("### Sample data")
st.dataframe(df.head(50), use_container_width=True)

# Data quality quick counts
st.markdown("### Quick Data Quality")
dq_cols = ["year", "month", "date", booking_col, delivered_col]
dq_info = {}
for c in dq_cols:
    if c and c in df.columns:
        non_null = int(df[c].notna().sum())
        dq_info[c] = {"non_null": non_null, "pct": round(100 * non_null / max(1, len(df)), 1)}
if dq_info:
    dq_df = pd.DataFrame.from_dict(dq_info, orient="index")
    st.table(dq_df)

# ---------- Analytics: Top-K ----------
st.markdown("## Quick Analytics (Top-K)")
with st.form("analytics_form", clear_on_submit=False):
    groupby = st.selectbox("Group by column", options=list(df.columns), index= df.columns.get_loc("section") if "section" in df.columns else 0)
    metric = st.selectbox("Metric (numeric)", options=[c for c in df.columns if c!=groupby], index=0)
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)
    run_analytics = st.form_submit_button("Run analytics")

if run_analytics:
    # try convert metric to numeric
    s = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    agg = df.copy()
    agg[metric] = s
    summary = agg.groupby(groupby)[metric].sum().sort_values(ascending=False).head(int(top_k))
    st.markdown(f"### Top {top_k} by `{metric}` grouped by `{groupby}`")
    # table
    st.dataframe(summary.reset_index().rename(columns={metric: f"{metric}_sum"}), use_container_width=True)
    # plot
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(summary))))
    summary.sort_values().plot.barh(ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel(groupby)
    ax.set_title(f"Top {top_k} {groupby} by {metric}")
    plt.tight_layout()
    st.pyplot(fig)
    # save outputs to outputs/final
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_out = OUT_DIR / f"summary-{metric}-by-{groupby}-{timestamp}.csv"
    png_out = OUT_DIR / f"{metric}_by_{groupby}-{timestamp}.png"
    summary.reset_index().rename(columns={metric: f"{metric}_sum"}).to_csv(csv_out, index=False)
    fig.savefig(png_out)
    st.success(f"Wrote summary CSV -> {csv_out.name} and plot -> {png_out.name}")

# ---------- Semantic QA UI ----------
st.markdown("## Semantic QA (RAG + LLM)")
st.write("Pick a demo question or type your own. The app will run the repo's `semantic_qa.py` which uses the RAG artifacts at outputs/rag.")

qa_col1, qa_col2 = st.columns([3,1])
question = qa_col1.text_area("Question", value=demo_q1, height=120)
k = qa_col2.number_input("k (evidence passages)", min_value=1, max_value=20, value=6)

run_qa = st.button("Run Semantic QA")

if run_qa:
    st.info("Running semantic QA — this will call `python src/semantic_qa.py` in the repo. Ensure your environment (embeddings/LLM keys) is set up.")
    # call semantic_qa.py via subprocess, capture output
    cmd = f'python src/semantic_qa.py --q "{question}" --rag_dir outputs/rag --k {k}'
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=False)
        st.text("=== STDOUT ===")
        st.text(proc.stdout)
        if proc.stderr:
            st.text("=== STDERR ===")
            st.text(proc.stderr)
        # find latest report in outputs/ (report-*.md)
        reports = sorted(list((ROOT / "outputs").glob("report-*.md")), key=lambda p: p.stat().st_mtime, reverse=True)
        evidence_files = sorted(list((ROOT / "outputs").glob("evidence-*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if reports:
            rpt = reports[0].read_text(encoding="utf8")
            st.markdown("### Generated Report (latest)")
            st.markdown(rpt)
            st.success(f"Report saved: {reports[0].name}")
        if evidence_files:
            ev = json.loads(evidence_files[0].read_text(encoding="utf8"))
            st.markdown("### Evidence (latest)")
            st.json(ev)
    except Exception as e:
        st.error(f"Failed to run semantic_qa.py: {e}")

# ---------- Pipeline runner ----------
if run_pipeline:
    st.warning("Running full pipeline (may take time). This calls `python src/langgraph_shim.py --config configs/config.yaml --auto-approve`.")
    cmd = f'python src/langgraph_shim.py --config configs/config.yaml --auto-approve'
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=False)
        st.text("=== PIPELINE STDOUT ===")
        st.text(proc.stdout)
        if proc.stderr:
            st.text("=== PIPELINE STDERR ===")
            st.text(proc.stderr)
        st.success("Pipeline run finished (check outputs/).")
    except Exception as e:
        st.error(f"Failed to run pipeline: {e}")

st.markdown("---")
st.caption("Designed to be data-agnostic. The app uses simple heuristics to find bookings/delivered columns and will work for similar datasets.")
