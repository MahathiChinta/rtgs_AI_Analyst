
"""
Run the ingest + clean_enhanced pipeline on multiple CSVs to demonstrate data-agnostic behaviour.

Usage:
python tests/test_on_files.py data/raw/tankers_reports_2024_3.csv data/raw/tankers_reports_2024_4.csv
"""
import sys
import subprocess
from pathlib import Path

def run_one(csv_path):
    csvp = Path(csv_path)
    out_clean = Path("data/cleaned") / f"test_cleaned_{csvp.stem}.csv"
    print("Running clean_enhanced on", csv_path)
    cmd = ["python", "src/clean_enhanced.py", "--in", str(csvp), "--out", str(out_clean), "--auto-approve"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR:", res.stderr)
        return False
    # check output exists
    if not out_clean.exists():
        print("Output not created:", out_clean)
        return False
    print("Success:", out_clean)
    return True

def main(args):
    all_ok = True
    for p in args:
        ok = run_one(p)
        all_ok = all_ok and ok
    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests failed.")
    return 0 if all_ok else 2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_on_files.py <csv1> <csv2> ...")
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1:]))
