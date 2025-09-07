
"""
Single-command runner:
 - cleans outputs (optional)
 - runs the orchestrator (ingest -> clean -> validate -> rag -> semantic_qa artifacts)
 - launches the interactive semantic CLI to accept your question

Usage:
  python src/run_agent.py --config configs/config.yaml --clean_outputs
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rm_outputs():
    import shutil
    for d in ["outputs", "traces"]:
        p = ROOT / d
        if p.exists():
            print("Removing", p)
            shutil.rmtree(p)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

def run_orchestrator(config_file: str):
    # run langgraph_shim.py (the orchestrator) with auto-approve so it moves without HITL
    cmd = [sys.executable, str(ROOT / "src" / "langgraph_shim.py"), "--config", config_file, "--auto-approve"]
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode

def launch_cli():
    cmd = [sys.executable, str(ROOT / "src" / "semantic_cli.py"), "--rag_dir", "outputs/rag"]
    print("Launching interactive CLI. Type your question when prompted.")
    subprocess.run(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--clean_outputs", action="store_true", help="Remove previous outputs/traces before run")
    args = p.parse_args()
    if args.clean_outputs:
        rm_outputs()
    rc = run_orchestrator(args.config)
    if rc != 0:
        print("Orchestrator returned non-zero rc:", rc)
        print("Check logs; aborting CLI launch.")
        sys.exit(rc)
    launch_cli()

if __name__ == "__main__":
    main()
