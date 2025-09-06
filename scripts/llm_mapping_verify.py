
"""
scripts/llm_mapping_verify.py

Run mapping verification on multiple CSVs to demonstrate data-agnostic behavior.
Usage:
  python scripts/llm_mapping_verify.py data/raw/file1.csv data/raw/file2.csv --auto-approve
"""
import argparse, subprocess, pathlib, sys

def run_one(csv_path, auto_approve=False):
    out = "data/cleaned/tmp_cleaned.csv"
    cmd = ["python", "src/clean_enhanced.py", "--in", csv_path, "--out", out]
    if auto_approve:
        cmd.append("--auto-approve")
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode

def main(files, auto_approve=False):
    for f in files:
        rc = run_one(f, auto_approve)
        print(f"Finished {f} -> rc={rc}")
        if rc != 0:
            print("Aborted on", f)
            break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--auto-approve", action="store_true")
    args = p.parse_args()
    main(args.files, auto_approve=args.auto_approve)
