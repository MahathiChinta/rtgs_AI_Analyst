#!/usr/bin/env python3
"""
clean_enhanced.py

Data-agnostic cleaning and standardization.

Inputs:
  --in  : path to input CSV (can be merged CSV from ingress)
  --out : path to write cleaned CSV (required)

Outputs (auto-created):
  - cleaned CSV (as specified by --out)
  - outputs/profiles/mapping-<ts>.json  (column type mapping + basic stats)
  - outputs/dq_report-<ts>.json         (data quality summary)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_column_type(series: pd.Series) -> str:
    """Return one of: numeric, datetime, categorical, text, boolean, unknown"""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    # numeric: ints/floats (with many unique)
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # attempt datetime parse on a sample
    sample = series.dropna().astype(str).head(200)
    if len(sample) > 0:
        success = 0
        for v in sample:
            try:
                dateparser.parse(v, fuzzy=False)
                success += 1
            except Exception:
                break
        if success >= max(1, len(sample) * 0.6):
            return "datetime"
    # categorical heuristics
    nunique = series.nunique(dropna=True)
    total = len(series)
    if nunique <= 50 or (nunique / max(1, total)) < 0.05:
        return "categorical"
    # fallback to text
    return "text"


def coerce_column(series: pd.Series, col_type: str) -> pd.Series:
    if col_type == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if col_type == "datetime":
        # try to parse; coerce failures to NaT
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if col_type == "boolean":
        # coerce common boolean forms
        return series.map(lambda x: True if str(x).strip().lower() in ("1", "true", "t", "yes", "y") else
                          (False if str(x).strip().lower() in ("0", "false", "f", "no", "n") else pd.NA))
    # categorical/text: keep as string but strip
    return series.astype(str).replace({"nan": pd.NA, "None": pd.NA})


def build_column_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile = {}
    for col in df.columns:
        s = df[col]
        typ = detect_column_type(s)
        non_null = s.dropna()
        stats = {
            "type_guess": typ,
            "count": int(len(s)),
            "non_null": int(non_null.shape[0]),
            "pct_missing": float(round((1 - non_null.shape[0] / max(1, len(s))) * 100, 3)),
            "n_unique": int(s.nunique(dropna=True)),
            "example_values": list(non_null.head(5).astype(str).tolist()),
        }
        # numeric extra stats
        if typ == "numeric":
            num = pd.to_numeric(s, errors="coerce").dropna()
            if len(num) > 0:
                stats.update({
                    "min": float(num.min()),
                    "max": float(num.max()),
                    "mean": float(num.mean()),
                    "std": float(num.std()) if len(num) > 1 else 0.0,
                })
        # datetime extra stats
        if typ == "datetime":
            dt = pd.to_datetime(s, errors="coerce").dropna()
            if len(dt) > 0:
                stats.update({
                    "min": dt.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "max": dt.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
        profile[col] = stats
    return profile


def main(infile: str, outfile: str):
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    prof_dir = out_dir / "profiles"
    safe_mkdir(prof_dir)
    safe_mkdir(out_dir)

    ts = TS()

    # --- load
    df = pd.read_csv(infile, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN", "nan", None])
    # treat empty strings as NaN
    df = df.replace({"": pd.NA})
    initial_rows = len(df)

    # --- column profiling (pre-coerce)
    pre_profile = build_column_profile(df)

    # --- determine column types and coerce
    coerced = {}
    for col in df.columns:
        typ = pre_profile[col]["type_guess"]
        coerced[col] = coerce_column(df[col], typ)

    dfc = pd.DataFrame(coerced)

    # --- create canonical id if none exists
    possible_id_cols = [c for c in dfc.columns if str(c).lower() in ("id", "uid", "rowid")]
    if len(possible_id_cols) == 0:
        # add stable row id
        dfc.insert(0, "row_id", [f"r{idx+1}" for idx in range(len(dfc))])
        generated_id = True
    else:
        # ensure uniqueness; if duplicates, still create row_id backup
        idcol = possible_id_cols[0]
        if dfc[idcol].is_unique:
            dfc.rename(columns={idcol: "row_id"}, inplace=True)
            generated_id = False
        else:
            dfc.insert(0, "row_id", [f"r{idx+1}" for idx in range(len(dfc))])
            generated_id = True

    # --- normalize datetimes and add year/month if available
    for col in list(dfc.columns):
        if pd.api.types.is_datetime64_any_dtype(dfc[col]):
            # ensure ISO format strings
            dfc[col] = pd.to_datetime(dfc[col], errors="coerce")
            # create helper columns
            dfc[f"{col}_iso"] = dfc[col].dt.strftime("%Y-%m-%d").where(dfc[col].notna(), pd.NA)
            dfc[f"{col}_year"] = dfc[col].dt.year.where(dfc[col].notna(), pd.NA)
            dfc[f"{col}_month"] = dfc[col].dt.month.where(dfc[col].notna(), pd.NA)

    # --- finalize: keep types friendly for CSV (dates as ISO strings)
    for col in dfc.columns:
        if pd.api.types.is_datetime64_any_dtype(dfc[col]):
            dfc[col] = dfc[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # numeric: keep numeric; categorical/text: convert to string but keep NaNs
        # pandas will represent pd.NA as empty in CSV by default; we keep it.

    # --- build mapping/profile after coercion
    post_profile = build_column_profile(dfc)

    # --- DQ checks (generic)
    dq = {"row_count": int(initial_rows), "generated_row_id": bool(generated_id), "issues": []}
    # check numeric columns for extreme missingness
    for col, meta in post_profile.items():
        if meta["pct_missing"] > 90:
            dq["issues"].append({"name": "high_missing", "col": col, "pct_missing": meta["pct_missing"]})
        if meta["type_guess"] == "numeric":
            if meta.get("n_unique", 0) <= 1:
                dq["issues"].append({"name": "low_variance_numeric", "col": col, "n_unique": meta.get("n_unique", 0)})
        if meta["type_guess"] == "categorical" and meta.get("n_unique", 0) == 0:
            dq["issues"].append({"name": "empty_categorical", "col": col})

    # write outputs
    cleaned_out = Path(outfile)
    cleaned_out.parent.mkdir(parents=True, exist_ok=True)
    # write cleaned CSV (use ISO strings for dates where present)
    dfc.to_csv(cleaned_out, index=False)

    mapping = {
        "generated_at": ts,
        "source_file": str(infile),
        "row_count": int(initial_rows),
        "generated_row_id": bool(generated_id),
        "pre_profile": pre_profile,
        "post_profile": post_profile,
    }
    mapping_path = prof_dir / f"mapping-{ts}.json"
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, default=str)

    dq_path = out_dir / f"dq_report-{ts}.json"
    with open(dq_path, "w", encoding="utf-8") as fh:
        json.dump(dq, fh, indent=2, default=str)

    print(f"Wrote cleaned CSV to {cleaned_out} ({initial_rows} rows)")
    print(f"Wrote mapping to {mapping_path}")
    print(f"Wrote DQ report to {dq_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path")
    p.add_argument("--out", dest="outfile", required=True, help="Output cleaned CSV path")
    args = p.parse_args()
    main(args.infile, args.outfile)
