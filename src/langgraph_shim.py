
"""
langgraph_shim.py - simple orchestrator that runs a sequence:
  researcher -> ingest -> clean_enhanced -> validate -> rag_builder -> semantic_qa

It does not require real LangGraph; acts as a small orchestrator for the pipeline.
Usage:
  python src/langgraph_shim.py --config configs/config.yaml [--auto-approve]
"""
import argparse, subprocess, yaml, os, sys
from pathlib import Path

def run_cmd(cmd, shell=False, check=True):
    print("RUN:", " ".join(cmd) if isinstance(cmd, list) else cmd)
    res = subprocess.run(cmd if isinstance(cmd, list) else cmd, shell=shell)
    if check and res.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")
    return res.returncode

def main(cfg_path, auto_approve=False):
    p = Path(cfg_path)
    if not p.exists():
        raise SystemExit("Config not found: " + cfg_path)
    cfg = yaml.safe_load(p.read_text())
    dataset_in = cfg.get("dataset", {}).get("infile", "data/cleaned/tankers_raw_merged.csv")
    cleaned_out = "data/cleaned/tankers_cleaned_enhanced.csv"
    # 1) researcher step (placeholder)
    print("[researcher] running...")
    files = list(Path("data/raw").glob("*.csv"))
    print(f"[researcher] files: {len(files)}")
    # 2) ingest (use src/ingest.py if exists)
    if Path("src/ingest.py").exists():
        run_cmd(["python", "src/ingest.py", "--input_dir", "data/raw", "--out", "data/cleaned/tankers_raw_merged.csv"])
    else:
        print("[ingest] fallback: expect merged CSV at data/cleaned/tankers_raw_merged.csv")
    # 3) clean_enhanced (HITL or auto)
    cmd = ["python", "src/clean_enhanced.py", "--in", "data/cleaned/tankers_raw_merged.csv", "--out", cleaned_out]
    if auto_approve:
        cmd.append("--auto-approve")
    run_cmd(cmd)
    # 4) validate
    if Path("src/validate.py").exists():
        run_cmd(["python", "src/validate.py", "--in", cleaned_out])
    # 5) rag builder (fallback)
    if Path("src/rag_builder_fallback.py").exists():
        run_cmd(["python", "src/rag_builder_fallback.py", "--in", cleaned_out, "--out_dir", cfg.get("rag", {}).get("out_dir", "outputs/rag"), "--model", cfg.get("rag", {}).get("embed_model", "all-MiniLM-L6-v2")])
    else:
        print("[rag] skip - rag_builder_fallback.py not found (ok for now)")
    # 6) semantic QA (run a single question if present)
    run_cfg = cfg.get("run", {})
    question = run_cfg.get("question")
    if question and Path("src/semantic_qa.py").exists():
        provider = cfg.get("llm", {}).get("provider", "manual")
        model = cfg.get("llm", {}).get("model", None)
        temp = cfg.get("llm", {}).get("temperature", 0.0)
        run_cmd(["python", "src/semantic_qa.py", "--q", question, "--rag_dir", cfg.get("rag", {}).get("out_dir","outputs/rag"), "--k", str(run_cfg.get("k",6)), "--provider", provider, "--llm_model", model, "--temp", str(temp)])
    print("[orchestrator] done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--auto-approve", action="store_true")
    args = p.parse_args()
    main(args.config, auto_approve=args.auto_approve)
