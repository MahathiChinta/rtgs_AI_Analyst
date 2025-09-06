"""
agent_pipeline.py
Top-level orchestrator for the RTGS agentic workflow.
Usage:
python src/agent_pipeline.py --config configs/config.yaml
"""
import argparse, subprocess, yaml, os, json
from pathlib import Path
from datetime import datetime

def run_cmd(cmd):
    print("RUN:", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def read_config(path):
    with open(path, "r", encoding="utf8") as fh:
        return yaml.safe_load(fh)

def human_approve(prompt):
    ans = input(f"{prompt} (y/n) > ").strip().lower()
    return ans in ("y","yes")

def main(config_path, auto_approve=False):
    cfg = read_config(config_path)
    # 1) ingestion (assume ingest script merges raw CSVs)
    infile = Path("data/cleaned/tankers_raw_merged.csv")
    if not infile.exists():
        cmd = f"python src/ingest.py --input_dir data/raw --out data/cleaned/tankers_raw_merged.csv"
        run_cmd(cmd)
    # 2) cleaning
    cleaned = Path(cfg["dataset"]["infile"])
    cmd_clean = f"python src/clean_enhanced.py --in data/cleaned/tankers_raw_merged.csv --out {cleaned}"
    run_cmd(cmd_clean)
    # 3) validation
    cmd_val = f"python src/validate.py --in {cleaned}"
    run_cmd(cmd_val)
    # read last validation file to decide
    vdir = Path("outputs")
    val_files = sorted(vdir.glob("validation-*.json"), key=lambda p: p.stat().st_mtime)
    if val_files:
        with open(val_files[-1], "r", encoding="utf8") as fh:
            val = json.load(fh)
        fails = [c for c in val.get("checks", []) if c.get("ok") is False]
        if fails:
            print("Validation flagged potential issues:")
            for f in fails:
                print("-", f)
            if not auto_approve:
                ok = human_approve("Validation has warnings. Approve continuing to RAG & LLM?")
                if not ok:
                    print("Aborting per user decision. Fix data or use --auto-approve to skip.")
                    return
    # 4) RAG build
    cmd_rag = f"python src/rag_builder_fallback.py --in {cleaned} --out_dir {cfg['rag']['out_dir']} --model {cfg['rag']['embed_model']}"
    run_cmd(cmd_rag)
    # 5) semantic QA (HITL approve)
    question = cfg["run"]["question"]
    print("\nReady to run semantic QA on question:\n", question)
    if not auto_approve:
        ok = human_approve("Approve running semantic QA with LLM?")
        if not ok:
            print("Aborting before LLM per user decision.")
            return
    cmd_qa = f"python src/semantic_qa.py --q \"{question}\" --rag_dir {cfg['rag']['out_dir']} --k {cfg['run']['k']} --provider {cfg['llm']['provider']} --llm_model {cfg['llm']['model']} --temp {cfg['llm']['temperature']}"
    run_cmd(cmd_qa)
    # 6) write trace
    trace = {"pipeline": "agentic_rtgs", "time": datetime.utcnow().isoformat(), "question": question}
    trace_dir = Path("traces"); trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"run-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(trace_path, "w", encoding="utf8") as fh:
        json.dump(trace, fh, indent=2, ensure_ascii=False)
    print("Pipeline complete. Trace written to", trace_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--auto-approve", action="store_true", help="skip HITL approvals")
    args = ap.parse_args()
    main(args.config, auto_approve=args.auto_approve)
