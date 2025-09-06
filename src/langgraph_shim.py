"""
src/langgraph_shim.py

LangGraph-style agentic workflow shim for RTGS project.

- Primary behavior: if `langgraph` is available, build a StateGraph with nodes:
  Researcher -> Ingester -> Cleaner -> Validator -> RAGBuilder -> QAAgent -> Reporter
- If langgraph is not installed or import fails, falls back to a safe orchestrator
  that calls your existing scripts (ingest.py, clean_enhanced.py, validate.py,
  rag_builder_fallback.py, semantic_qa.py) via subprocess or direct imports.
- Data-agnostic: uses src.schema_infer.infer_schema to auto-detect column roles
  and writes mapping to outputs/profiles/*.json.
- HITL: after validation, graph pauses and asks for approval unless --auto-approve.
- Produces a trace JSON: traces/run-<ts>.json
- Run: python src/langgraph_shim.py --config configs/agents.yaml [--auto-approve]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import importlib

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Try to import our helpers
try:
    from src import schema_infer  # type: ignore
except Exception:
    schema_infer = None  # we'll still work without it (fallback)

# Paths
CONFIG_DEFAULT = ROOT / "configs" / "agents.yaml"
TRACE_DIR = ROOT / "traces"
OUTPUTS = ROOT / "outputs"
PROFILES = OUTPUTS / "profiles"
DATA_CLEANED = ROOT / "data" / "cleaned"
DATA_RAW = ROOT / "data" / "raw"

# timestamp helper
def ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


# Utilities
def run_cmd(cmd: str):
    print("RUN:", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def save_trace(trace: dict):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    fn = TRACE_DIR / f"run-{ts()}.json"
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2)
    print("[trace] saved to", fn)
    return fn


# Thin wrappers around existing scripts (prefer direct import, fallback to subprocess)
def ingest_files(input_dir: str, out_csv: str):
    """
    Try to call src.ingest.main if available, else merge CSVs in input_dir.
    """
    try:
        ingest_mod = importlib.import_module("src.ingest")
        if hasattr(ingest_mod, "main"):
            # ingest.main expects argv-like list
            ingest_mod.main(["--input_dir", input_dir, "--out", out_csv])
            return out_csv
    except Exception:
        pass

    # Fallback: naive CSV concat
    import pandas as pd
    p = Path(input_dir)
    files = list(p.glob("*.csv"))
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    merged = pd.concat(dfs, ignore_index=True)
    PATH = Path(out_csv)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PATH, index=False)
    print("[ingest] fallback merged", len(dfs), "files ->", PATH)
    return str(PATH)


def clean_enhanced(in_csv: str, out_csv: str, mapping_out: str, dq_out: str):
    """
    Try to call clean_enhanced.main, otherwise run a lightweight canonicalization using schema_infer.
    """
    try:
        clean_mod = importlib.import_module("src.clean_enhanced")
        if hasattr(clean_mod, "main"):
            clean_mod.main(["--in", in_csv, "--out", out_csv])
            # try to write mapping/dq if created by script
            return out_csv
    except Exception:
        pass

    # Fallback: use schema_infer to canonicalize and save mapping + dq
    import pandas as pd
    df = pd.read_csv(in_csv)
    if schema_infer:
        mapping = schema_infer.infer_schema(df)
    else:
        # minimal mapping: canonicalize column names
        mapping = {"columns": {}}
        for c in df.columns:
            mapping["columns"][c] = {"canon": c.lower().strip().replace(" ", "_"), "pct_missing": float(df[c].isna().mean() * 100)}
    # rename columns
    rename = {c: mapping["columns"][c]["canon"] for c in df.columns}
    df = df.rename(columns=rename)
    # ensure id and date exist
    if "id" not in df.columns:
        df["id"] = df.apply(lambda r: hash(tuple(r.values)), axis=1)
    if "date" not in df.columns and ("year" in df.columns and "month" in df.columns):
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01", errors="coerce")
    # write outputs
    Path(mapping_out).parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    Path(dq_out).parent.mkdir(parents=True, exist_ok=True)
    dq = {"rows": len(df), "columns": {c: {"pct_missing": mapping["columns"][c]["pct_missing"]} for c in mapping["columns"]}}
    with open(dq_out, "w", encoding="utf-8") as fh:
        json.dump(dq, fh, indent=2)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[clean] fallback wrote", out_csv, mapping_out, dq_out)
    return out_csv


def validate_csv(cleaned_csv: str, out_validation: str):
    """
    Run validate.py if present; otherwise produce a minimal validation JSON.
    """
    try:
        val_mod = importlib.import_module("src.validate")
        if hasattr(val_mod, "main"):
            val_mod.main(["--in", cleaned_csv])
            # The script writes outputs; best-effort return
            return out_validation
    except Exception:
        pass

    # fallback minimal validation
    import pandas as pd
    df = pd.read_csv(cleaned_csv)
    validation = {"rows": len(df), "warnings": []}
    for c in df.columns:
        pct_missing = float(df[c].isna().mean() * 100)
        if pct_missing > 50:
            validation["warnings"].append({"col": c, "pct_missing": pct_missing})
    Path(out_validation).parent.mkdir(parents=True, exist_ok=True)
    with open(out_validation, "w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2)
    print("[validate] fallback wrote", out_validation)
    return out_validation


def build_rag(cleaned_csv: str, out_dir: str, emb_model: str = "all-MiniLM-L6-v2"):
    """
    Call rag_builder_fallback.py or fallback to simple passages.jsonl + embeddings (if sentence-transformers installed).
    """
    try:
        rag_mod = importlib.import_module("src.rag_builder_fallback")
        if hasattr(rag_mod, "main"):
            rag_mod.main(["--in", cleaned_csv, "--out_dir", out_dir, "--model", emb_model])
            return out_dir
    except Exception:
        pass

    # fallback: simple passages.jsonl
    import pandas as pd, json
    df = pd.read_csv(cleaned_csv)
    pdir = Path(out_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    passages = []
    for i, row in df.iterrows():
        text = " | ".join([f"{k}: {row[k]}" for k in df.columns if k != "id"])
        passages.append({"id": str(i), "text": text, "meta": {"row_id": i}})
    with open(pdir / "passages.jsonl", "w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p) + "\n")
    print("[rag] fallback wrote passages.jsonl (count {})".format(len(passages)))
    return str(pdir)


def run_semantic_qa(question: str, rag_dir: str, provider: str, llm_model: str, temp: float = 0.0):
    """
    Call semantic_qa.py to perform retrieval+LLM answer. If it cannot call programmatically,
    prints a manual prompt (the script already does that).
    """
    try:
        sa = importlib.import_module("src.semantic_qa")
        if hasattr(sa, "main"):
            sa.main(question, rag_dir=rag_dir, k=6, llm_provider=provider, llm_model=llm_model, temp=temp)
            return True
    except Exception:
        pass

    # fallback run as subprocess
    cmd = f'python src/semantic_qa.py --q "{question}" --rag_dir {rag_dir} --k 6 --provider {provider} --llm_model {llm_model} --temp {temp}'
    run_cmd(cmd)
    return True


# Try to import langgraph (official); if not present we'll run the fallback orchestrator
try:
    import langgraph as lg  # type: ignore
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


def orchestrator(cfg: dict, auto_approve: bool = False):
    """
    Fallback orchestrator (non-LangGraph) implementing the same multi-agent flow.
    """
    # Ensure outputs exist
    PROFILES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DATA_CLEANED.mkdir(parents=True, exist_ok=True)

    # 1) Researcher: dataset_files from cfg
    dataset_files = cfg.get("dataset_files", [])
    if not dataset_files:
        # If none provided, assume all CSVs in data/raw
        dataset_files = [str(p) for p in (DATA_RAW).glob("*.csv")]
    print("[researcher] files:", len(dataset_files))

    # 2) Ingester: merge -> merged csv
    merged = str(DATA_CLEANED / f"merged-{ts()}.csv")
    # if dataset_files are in a directory, pick dir of first
    input_dir = str(Path(dataset_files[0]).parent) if dataset_files else str(DATA_RAW)
    ingest_files(input_dir, merged)

    # 3) Clean
    cleaned = str(DATA_CLEANED / f"cleaned-{ts()}.csv")
    mapping = str(PROFILES / f"mapping-{ts()}.json")
    dq_report = str(OUTPUTS / f"dq_report-{ts()}.json")
    clean_enhanced(merged, cleaned, mapping, dq_report)

    # 4) Validate (HITL)
    validation_out = str(OUTPUTS / f"validation-{ts()}.json")
    validate_csv(cleaned, validation_out)
    # Read validation warnings and request approval if needed
    try:
        with open(validation_out, "r", encoding="utf-8") as fh:
            v = json.load(fh)
            warnings = v.get("warnings", [])
    except Exception:
        warnings = []
    if warnings and not auto_approve:
        print("[supervisor] validation warnings:", warnings)
        ans = input("Approve continuing to RAG & LLM? (y/n) > ").strip().lower()
        if ans != "y":
            print("Aborting as per user.")
            return

    # 5) RAG builder
    rag_out = str(OUTPUTS / "rag" / f"run-{ts()}")
    build_rag(cleaned, rag_out, emb_model=cfg.get("emb_model", "all-MiniLM-L6-v2"))

    # 6) QA
    question = cfg.get("question", "Which localities show the largest unmet tanker demand? Provide short policy recommendations.")
    provider = cfg.get("provider", "gemini")
    llm_model = cfg.get("llm_model", "gemini-2.5-pro")
    run_semantic_qa(question, rag_out, provider, llm_model, temp=float(cfg.get("temp", 0.0)))

    # 7) Trace
    trace = {
        "run_ts": ts(),
        "merged": merged,
        "cleaned": cleaned,
        "mapping": mapping,
        "dq_report": dq_report,
        "rag_dir": rag_out
    }
    save_trace(trace)
    print("[orchestrator] done.")


def build_langgraph_graph(cfg: dict, auto_approve: bool = False):
    """
    Build a LangGraph StateGraph that models the same flow.
    This is written defensively because langgraph APIs vary; we attempt a canonical approach.
    """
    # Try to create a minimal graph using langgraph
    # Note: exact API names may differ by version; this code uses a conservative style.
    try:
        from langgraph.graph import StateGraph, Node  # type: ignore
    except Exception as e:
        print("LangGraph import failed or API not found:", repr(e))
        raise

    graph = StateGraph(name="rtgs-agent-graph")

    # Node wrappers - nodes call our functions
    def node_researcher(_inputs):
        dataset_files = cfg.get("dataset_files", [])
        if not dataset_files:
            dataset_files = [str(p) for p in (DATA_RAW).glob("*.csv")]
        return {"dataset_files": dataset_files}

    def node_ingester(inputs):
        files = inputs["dataset_files"]
        input_dir = str(Path(files[0]).parent) if files else str(DATA_RAW)
        merged = str(DATA_CLEANED / f"merged-{ts()}.csv")
        ingest_files(input_dir, merged)
        return {"merged": merged}

    def node_cleaner(inputs):
        merged = inputs["merged"]
        cleaned = str(DATA_CLEANED / f"cleaned-{ts()}.csv")
        mapping = str(PROFILES / f"mapping-{ts()}.json")
        dq = str(OUTPUTS / f"dq_report-{ts()}.json")
        clean_enhanced(merged, cleaned, mapping, dq)
        return {"cleaned": cleaned, "mapping": mapping, "dq": dq}

    def node_validator(inputs):
        cleaned = inputs["cleaned"]
        validation_out = str(OUTPUTS / f"validation-{ts()}.json")
        validate_csv(cleaned, validation_out)
        return {"validation": validation_out}

    def node_rag(inputs):
        cleaned = inputs["cleaned"]
        rag_out = str(OUTPUTS / "rag" / f"run-{ts()}")
        build_rag(cleaned, rag_out, emb_model=cfg.get("emb_model", "all-MiniLM-L6-v2"))
        return {"rag_dir": rag_out}

    def node_qa(inputs):
        rag_dir = inputs["rag_dir"]
        question = cfg.get("question")
        run_semantic_qa(question, rag_dir, cfg.get("provider", "gemini"), cfg.get("llm_model", "gemini-2.5-pro"), temp=float(cfg.get("temp", 0.0)))
        return {"report": "outputs/report-<ts>.md"}

    # Register nodes (API may vary)
    graph.add_node("researcher", node_researcher)
    graph.add_node("ingester", node_ingester, inputs=["researcher"])
    graph.add_node("cleaner", node_cleaner, inputs=["ingester"])
    graph.add_node("validator", node_validator, inputs=["cleaner"])
    graph.add_node("rag", node_rag, inputs=["cleaner"])
    graph.add_node("qa", node_qa, inputs=["rag"])

    # Add a simple orchestration runner
    print("[langgraph] built graph with nodes: researcher->ingester->cleaner->validator->rag->qa")
    # Execute graph in a simple linear fashion (since API variants exist)
    state = {}
    state.update(node_researcher({}))
    state.update(node_ingester(state))
    state.update(node_cleaner(state))
    # validation with HITL
    v = node_validator(state)
    warnings = []
    try:
        with open(v["validation"], "r", encoding="utf-8") as fh:
            vv = json.load(fh)
            warnings = vv.get("warnings", [])
    except Exception:
        pass
    if warnings and not auto_approve:
        print("[langgraph-supervisor] validation warnings:", warnings)
        resp = input("Approve continuing to RAG & LLM? (y/n) > ").strip().lower()
        if resp != "y":
            print("Aborting as per user.")
            return
    # continue
    state.update(node_rag(state))
    state.update(node_qa(state))
    # trace
    trace = {"run_ts": ts(), "state": state}
    save_trace(trace)
    print("[langgraph] run complete.")


def load_config(path: str):
    try:
        import yaml
    except Exception:
        raise RuntimeError("PyYAML required; pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_DEFAULT))
    p.add_argument("--auto-approve", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    # merge with defaults
    cfg.setdefault("emb_model", "all-MiniLM-L6-v2")
    cfg.setdefault("provider", "gemini")
    cfg.setdefault("llm_model", "gemini-2.5-pro")
    cfg.setdefault("k", 6)
    cfg.setdefault("temp", 0.0)
    cfg.setdefault("question", "Which localities show the largest unmet tanker demand? Provide short policy recommendations.")

    if HAS_LANGGRAPH:
        try:
            build_langgraph_graph(cfg, auto_approve=args.auto_approve)
            return
        except Exception as e:
            print("LangGraph execution failed, falling back to orchestrator:", repr(e))

    # fallback orchestrator
    orchestrator(cfg, auto_approve=args.auto_approve)


if __name__ == "__main__":
    main()
