"""
ingest.py
Merge CSVs in input_dir producing a single merged CSV + profile.

Usage:
  python src/ingest.py --input_dir data/raw --out data/cleaned/tankers_raw_merged.csv
  python src/ingest.py --input_dir data/raw --out data/cleaned/tankers_raw_merged.csv --require-columns noofbookings,delivered
"""
import argparse, csv, json
from pathlib import Path
import pandas as pd
from datetime import datetime

def profile_df(df):
    prof = {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample": df.head(5).to_dict(orient="records")
    }
    return prof

def detect_file_has_columns(path: Path, required_cols):
    try:
        # read small sample to inspect columns cheaply
        df_sample = pd.read_csv(path, nrows=5, dtype=str)
        cols = [c.lower().strip() for c in df_sample.columns]
        for rc in required_cols:
            if rc.lower().strip() in cols:
                return True
        return False
    except Exception:
        return False

def main(input_dir, out, require_columns=None):
    input_dir = Path(input_dir)
    out = Path(out)
    files = sorted([p for p in input_dir.glob("*.csv")])
    if not files:
        print("No CSVs in", input_dir)
        return

    require_cols = []
    if require_columns:
        require_cols = [c.strip() for c in require_columns.split(",") if c.strip()]

    dfs = []
    used = []
    skipped = []
    for f in files:
        try:
            if require_cols:
                ok = detect_file_has_columns(f, require_cols)
                if not ok:
                    skipped.append(f.name)
                    print(f"Skipping {f.name} (does not contain any of: {require_cols})")
                    continue
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            df["_source_file"] = f.name
            dfs.append(df)
            used.append(f.name)
            print("Loaded", f.name, "->", len(df), "rows")
        except Exception as e:
            print("Failed to load", f, e)

    if not dfs:
        print("No files matched requirements; nothing to merge.")
        return

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    prof = profile_df(merged)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prof_path = Path("outputs/profiles") / f"profile-{ts}.json"
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prof_path, "w", encoding="utf8") as fh:
        json.dump(prof, fh, indent=2, ensure_ascii=False)

    print(f"Wrote merged CSV to {out} ({len(merged)} rows)")
    print(f"Wrote profile to {prof_path}")
    if skipped:
        print("Skipped files:", skipped)
    if used:
        print("Used files:", used)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require-columns", default=None,
                   help="Comma-separated list of columns; only CSVs containing at least one will be merged")
    args = p.parse_args()
    main(args.input_dir, args.out, require_columns=args.require_columns)
