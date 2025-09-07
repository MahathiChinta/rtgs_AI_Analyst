import streamlit as st
import subprocess, shlex, json, glob, os, textwrap
from pathlib import Path
st.set_page_config(page_title="RTGS Agent Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs"
RAG_DIR = ROOT / "outputs" / "rag"

st.title("RTGS-style AI Analyst — Dashboard")

st.markdown("**Quick analytics & latest generated artifacts**")

# 1) Show latest plot (png) and csv (if present)
plot_files = sorted(PLOTS_DIR.glob("*.png"), key=os.path.getmtime, reverse=True) if PLOTS_DIR.exists() else []
csv_files = sorted(PLOTS_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True) if PLOTS_DIR.exists() else []

if plot_files:
    st.subheader("Latest plot")
    st.image(str(plot_files[0]), use_column_width=True)
else:
    st.info("No plots found in outputs/plots/. Run the pipeline to generate quick analytics.")

if csv_files:
    st.subheader("Latest summary CSV (top rows)")
    df_path = csv_files[0]
    try:
        import pandas as pd
        df = pd.read_csv(df_path)
        st.dataframe(df.head(20))
        st.markdown(f"Download CSV: [{df_path.name}](./{df_path.relative_to(ROOT)})")
    except Exception as e:
        st.write("Could not load CSV:", e)

st.markdown("---")

# 2) Show latest report-*.md (LLM semantic QA report)
report_files = sorted(REPORTS_DIR.glob("report-*.md"), key=os.path.getmtime, reverse=True)
if report_files:
    st.subheader("Latest semantic QA report")
    rpt = report_files[0].read_text(encoding="utf-8")
    st.markdown(rpt)
else:
    st.info("No semantic QA reports found (report-*.md). Run semantic_qa.py first.")

st.markdown("---")

# 3) One-shot question box (non-interactive run)
st.subheader("Ask a question (one-shot)")
question = st.text_area("Type your question for the dataset (one question at a time):", height=100)
k = st.number_input("Number of evidence passages to retrieve (k):", min_value=1, max_value=20, value=6)
model = st.text_input("LLM model (use GEMINI model from config or leave blank):", value="")
submit = st.button("Run semantic Q&A")

if submit:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        st.info("Running semantic QA. This runs your repo script `semantic_qa.py` and will create evidence+report files in outputs/.")
        # Build command
        model_arg = f'--llm_model "{model}"' if model else ""
        cmd = f'python src/semantic_qa.py --q "{question}" --rag_dir outputs/rag --k {k} {model_arg} --temp 0.0'
        st.code(cmd)
        try:
            proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True, timeout=120)
            st.success("Completed.")
            st.text(proc.stdout)
            if proc.stderr:
                st.error(proc.stderr)
        except subprocess.CalledProcessError as e:
            st.error(f"semantic_qa failed: {e}\n{e.stderr}")
        except subprocess.TimeoutExpired:
            st.error("semantic_qa timed out (120s). Try a shorter run or run directly on shell.")

        # show newest report automatically
        rpt_list = sorted(REPORTS_DIR.glob("report-*.md"), key=os.path.getmtime, reverse=True)
        if rpt_list:
            st.markdown("### Latest generated report")
            st.markdown(rpt_list[0].read_text(encoding="utf-8"))
        else:
            st.warning("No report generated.")

st.markdown("---")
st.info("Tip: run the full pipeline first: `python src/langgraph_shim.py --config configs/config.yaml --auto-approve` then open this Streamlit dashboard.")
