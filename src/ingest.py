#!/usr/bin/env python3
"""
ingest.py
- Merge monthly CSVs from data/raw into a single merged CSV.
- Produce a tiny profile JSON in outputs/profiles/.

This is intentionally simple and robust for the buildathon.
"""
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def discover_csvs(input_dir):
    p = Path(input_dir)
    files = sorted([str(f) for f in p.glob("**/*.csv")])
    return files

def load_and_standardize(path):
    # Read as text to avoid dtype surprises; caller will clean later
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "N/A", None])
    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def simple_profile(df):
    prof = {}
    prof['rows'] = int(len(df))
    prof['columns'] = {col: {
        "dtype": str(df[col].dtype),
        "n_missing": int(df[col].isna().sum()) if df[col].dtype != 'O' else int((df[col]=='').sum()),
    } for col in df.columns}
    return prof

def main(input_dir, out):
    files = discover_csvs(input_dir)
    if not files:
        print(f"No CSV files found in {input_dir}.")
        return
    frames = []
    for f in files:
        try:
            df = load_and_standardize(f)
            df['_source_file'] = Path(f).name
            frames.append(df)
            print(f"Loaded {Path(f).name} -> {len(df)} rows")
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    merged = pd.concat(frames, ignore_index=True, sort=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote merged CSV to {out} ({len(merged)} rows)")
    prof = simple_profile(merged)
    prof_path = Path("outputs/profiles")
    prof_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prof_file = prof_path / f"profile-{ts}.json"
    with open(prof_file, "w", encoding="utf8") as fh:
        json.dump(prof, fh, indent=2)
    print(f"Wrote profile to {prof_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--out", default="data/cleaned/tankers_raw_merged.csv")
    args = parser.parse_args()
    main(args.input_dir, args.out)
