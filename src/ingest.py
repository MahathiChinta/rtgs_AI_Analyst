
"""
ingest.py
Merge all CSVs in input_dir and produce a single merged CSV and a small profile.
Usage:
  python src/ingest.py --input_dir data/raw --out data/cleaned/tankers_raw_merged.csv
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

def main(input_dir, out):
    input_dir = Path(input_dir)
    out = Path(out)
    files = sorted([p for p in input_dir.glob("*.csv")])
    if not files:
        print("No CSVs in", input_dir)
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            df["_source_file"] = f.name
            dfs.append(df)
            print("Loaded", f.name, "->", len(df), "rows")
        except Exception as e:
            print("Failed to load", f, e)

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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.input_dir, args.out)
