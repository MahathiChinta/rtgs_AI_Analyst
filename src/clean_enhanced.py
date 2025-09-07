#!/usr/bin/env python3
"""
clean_enhanced.py

Enhanced cleaning + mapping verification (HITL) for the RTGS pipeline.

Writes mapping, profile, and DQ report into outputs/final/.
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import pandas as pd

try:
    from src import llm_adapter as llm_adapter_pkg
except Exception:
    import llm_adapter as llm_adapter_pkg

TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def infer_and_fix_dates_and_counts(df: pd.DataFrame, filename_hint_col: str = "_source_file") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diag = {"found_cols": list(df.columns), "mapped": {}, "warnings": [], "sample_vals": {}}

    for c in df.columns:
        nonnull = df[c].dropna()
        diag["sample_vals"][c] = str(nonnull.iloc[0]) if len(nonnull) > 0 else None

    col_lower_map = {c.lower(): c for c in df.columns}

    def choose_col(patterns):
        for p in patterns:
            for k_lower, orig in col_lower_map.items():
                if p in k_lower:
                    return orig
        return None

    # bookings
    bookings_col = choose_col(["noofbookings", "bookings", "no_of_bookings", "number_of_bookings",
                               "booked", "requests", "totservices", "billdservices"])
    if bookings_col:
        diag["mapped"]["noofbookings"] = bookings_col
        df[bookings_col] = pd.to_numeric(df[bookings_col], errors="coerce")
        df["noofbookings"] = df.get("noofbookings", df[bookings_col])
    else:
        diag["warnings"].append("No obvious bookings column found")

    # delivered
    delivered_col = choose_col(["delivered", "delv", "supplied", "fulfilled", "supply"])
    if delivered_col:
        diag["mapped"]["delivered"] = delivered_col
        df[delivered_col] = pd.to_numeric(df[delivered_col], errors="coerce")
        df["delivered"] = df.get("delivered", df[delivered_col])
    else:
        diag["warnings"].append("No obvious delivered column found")

    # date column
    date_col = choose_col(["date", "booking_date", "req_date", "request_date", "dt"])
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        diag["mapped"]["date_from"] = date_col

    # year + month
    year_col = choose_col(["year"])
    month_col = choose_col(["month", "mon"])
    if year_col and month_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        mask = df[year_col].notna() & df[month_col].notna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, year_col].astype(int).astype(str) + "-" +
                df.loc[mask, month_col].astype(int).astype(str).str.zfill(2) + "-01",
                errors="coerce"
            )
            diag["mapped"]["date_from_year_month"] = (year_col, month_col)

    # fallback: extract year/month from filename
    if ("date" not in df.columns) or df["date"].isna().all():
        if filename_hint_col in df.columns:
            def extract_ym(s):
                if not isinstance(s, str):
                    return (None, None)
                m = re.search(r"(20\d{2})[^\d]?(0?[1-9]|1[0-2])", s)
                if m:
                    return int(m.group(1)), int(m.group(2))
                return (None, None)

            ym = df[filename_hint_col].apply(lambda s: extract_ym(str(s)))
            df["__src_year"] = ym.apply(lambda t: t[0])
            df["__src_month"] = ym.apply(lambda t: t[1])
            mask2 = df["__src_year"].notna() & df["__src_month"].notna()
            if mask2.any():
                df.loc[mask2, "date"] = pd.to_datetime(
                    df.loc[mask2, "__src_year"].astype(int).astype(str) + "-" +
                    df.loc[mask2, "__src_month"].astype(int).astype(str).str.zfill(2) + "-01",
                    errors="coerce"
                )
                diag["mapped"]["date_from_source_file"] = True

    # helpers
    if "date" in df.columns:
        df["month_iso"] = df["date"].dt.strftime("%Y-%m")
        df["year"] = df.get("year", df["date"].dt.year)
        df["month"] = df.get("month", df["date"].dt.month)

    # numeric coercion
    for c in ["noofbookings", "delivered", "year", "month"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop temp
    for c in ["__src_year", "__src_month"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    diag["counts"] = {
        "rows": len(df),
        "date_nonnull": int(df.get("date").notna().sum() if "date" in df else 0),
        "bookings_nonnull": int(df.get("noofbookings").notna().sum() if "noofbookings" in df else 0),
        "delivered_nonnull": int(df.get("delivered").notna().sum() if "delivered" in df else 0)
    }
    return df, diag


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def call_mapping_verification_llm(mapping: Dict[str, Any], sample: Dict[str, Any], auto_approve: bool = False) -> Tuple[bool, Dict]:
    lines = [
        "You are a schema-mapping assistant.",
        "Reply in JSON: {\"ok\": bool, \"suggestions\": [{\"raw\":..., \"suggested\":..., \"reason\":...}]}",
        "",
        "Raw columns:"
    ]
    for k, v in sample.items():
        lines.append(f"- {k}: {v}")
    lines.append("Mapping:")
    for k, v in mapping.items():
        lines.append(f"- {k} <- {v}")
    prompt = "\n".join(lines)

    try:
        llm_out = llm_adapter_pkg.call_llm(prompt=prompt, model=None, temperature=0.0, max_tokens=512)
        parsed = json.loads(llm_out.strip())
    except Exception as e:
        return False, {"ok": False, "error": str(e)}

    if auto_approve:
        return True, parsed
    return bool(parsed.get("ok")), parsed


def main(infile: str, out: str, auto_approve: bool = False):
    pin = Path(infile)
    pout = Path(out)
    if not pin.exists():
        raise SystemExit(f"File not found: {pin}")

    print("[clean_enhanced] loading", infile)
    df = pd.read_csv(pin)
    df_fixed, diag = infer_and_fix_dates_and_counts(df)

    mapping = {
        "date": diag.get("mapped", {}).get("date_from") or diag.get("mapped", {}).get("date_from_year_month") or ("_source_file" if diag.get("mapped", {}).get("date_from_source_file") else None),
        "noofbookings": diag.get("mapped", {}).get("noofbookings"),
        "delivered": diag.get("mapped", {}).get("delivered")
    }

    sample_row = {c: (df_fixed[c].dropna().astype(str).iloc[0] if df_fixed[c].notna().any() else None) for c in df_fixed.columns}

    final_dir = Path("outputs/final")
    final_dir.mkdir(parents=True, exist_ok=True)

    mapping_profile_path = final_dir / f"mapping-{TS()}.json"
    write_json({"mapping": mapping, "diag": diag, "sample_row": sample_row}, mapping_profile_path)
    print("Wrote mapping to", mapping_profile_path)

    print("[clean_enhanced] running mapping verification...")
    approved, _ = call_mapping_verification_llm(mapping, sample_row, auto_approve=auto_approve)
    if not approved and not auto_approve:
        raise SystemExit("Mapping not approved")

    df_fixed.to_csv(pout, index=False)
    print(f"Wrote cleaned CSV to {pout} ({len(df_fixed)} rows)")

    dq_path = final_dir / f"dq_report-{TS()}.json"
    write_json({"ts": TS(), "counts": diag.get("counts"), "warnings": diag.get("warnings")}, dq_path)
    print("Wrote DQ report to", dq_path)

    profile_path = final_dir / f"profile-{TS()}.json"
    write_json({"ts": TS(), "mapping_profile": mapping, "counts": diag.get("counts")}, profile_path)
    print("Wrote profile to", profile_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--auto-approve", action="store_true")
    args = p.parse_args()
    main(args.infile, args.out, args.auto_approve)
