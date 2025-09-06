#!/usr/bin/env python3
"""
validate.py

Generic validation checks for a cleaned CSV. Produces outputs/validation-<ts>.json
and prints a short human-readable summary.

Usage:
  python src/validate.py --in data/cleaned/tankers_cleaned_enhanced.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Timestamp helper (keep simple)
TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# Columns with these suffixes are considered "helper/generated" columns and should not
# produce high-noise warnings in the human summary. We still produce full diagnostics
# for them in the JSON output.
HELPER_SUFFIXES = (
    "_iso",
    "_year",
    "_month",
    "_month_year",
    "_year_month",
    "month_iso",
    "month_year",
    "year_year",
    "year_month",
)


def numeric_checks(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()
    return {
        "pct_missing": float(round((1 - len(non_null) / max(1, len(s))) * 100, 4)),
        "min": float(non_null.min()) if len(non_null) > 0 else None,
        "max": float(non_null.max()) if len(non_null) > 0 else None,
        "mean": float(non_null.mean()) if len(non_null) > 0 else None,
        "std": float(non_null.std()) if len(non_null) > 1 else None,
        "n_unique": int(series.nunique(dropna=True)),
    }


def datetime_checks(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_datetime(series, errors="coerce")
    non_null = s.dropna()
    return {
        "pct_missing": float(round((1 - len(non_null) / max(1, len(s))) * 100, 4)),
        "min": non_null.min().strftime("%Y-%m-%dT%H:%M:%SZ") if len(non_null) > 0 else None,
        "max": non_null.max().strftime("%Y-%m-%dT%H:%M:%SZ") if len(non_null) > 0 else None,
        "n_unique": int(series.nunique(dropna=True)),
    }


def categorical_checks(series: pd.Series) -> Dict[str, Any]:
    non_null = series.dropna().astype(str)
    uniques = non_null.value_counts().to_dict()
    top_k = dict(list(uniques.items())[:10])
    return {
        "pct_missing": float(round((1 - len(non_null) / max(1, len(series))) * 100, 4)),
        "n_unique": int(series.nunique(dropna=True)),
        "top_values": top_k,
    }


def _is_number(s: str) -> bool:
    try:
        float(str(s))
        return True
    except Exception:
        return False


def generate_validation_report(csv_path: str) -> Dict[str, Any]:
    # Load CSV defensively
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN", "nan", None])
    # normalize empty strings -> pd.NA
    df = df.replace({"": pd.NA})
    report = {"generated_at": TS(), "file": str(csv_path), "row_count": int(len(df)), "columns": {}, "warnings": []}

    for col in df.columns:
        series = df[col]
        # Guess type: numeric / datetime / categorical
        guessed_type = None
        checks = {}

        # numeric heuristic
        if pd.api.types.is_numeric_dtype(series) or all(_is_number(x) for x in series.dropna().astype(str).head(200)):
            checks = numeric_checks(series)
            guessed_type = "numeric"
        else:
            # try parse sample as datetime (without deprecated arg)
            try:
                parsed = pd.to_datetime(series.dropna().astype(str).head(200), errors="coerce")
                notnull = parsed.notna().sum()
                if notnull >= max(1, int(0.6 * len(parsed))):
                    checks = datetime_checks(series)
                    guessed_type = "datetime"
                else:
                    checks = categorical_checks(series)
                    guessed_type = "categorical"
            except Exception:
                checks = categorical_checks(series)
                guessed_type = "categorical"

        report["columns"][col] = {"guessed_type": guessed_type, "checks": checks}

        # Add warnings to human summary, but skip noisy helper columns
        is_helper = any(col.endswith(suf) or col.startswith(suf.strip("_")) for suf in HELPER_SUFFIXES)
        pct_missing = checks.get("pct_missing", 0)

        if (pct_missing > 90) and (not is_helper):
            report["warnings"].append({"col": col, "issue": "high_missing", "pct_missing": pct_missing})
        if guessed_type == "numeric" and (checks.get("n_unique", 0) <= 1) and (not is_helper):
            report["warnings"].append({"col": col, "issue": "low_variance_numeric", "n_unique": checks.get("n_unique", 0)})

    return report


def main(infile: str, outpath: str = None):
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    safe_mkdir(out_dir)
    ts = TS()
    report = generate_validation_report(infile)
    out_path = Path(outpath) if outpath else out_dir / f"validation-{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    # human readable summary
    print(f"Wrote validation results to {out_path}")
    print("Summary warnings:")
    if len(report.get("warnings", [])) == 0:
        print(" - no high-impact warnings (helper columns excluded)")
    else:
        for w in report.get("warnings", [])[:20]:
            print(" -", w)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="Cleaned CSV input")
    p.add_argument("--out", dest="outpath", required=False, help="Validation JSON output (optional)")
    args = p.parse_args()
    main(args.infile, args.outpath)
