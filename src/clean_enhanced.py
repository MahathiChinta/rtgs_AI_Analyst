#!/usr/bin/env python3
"""
clean_enhanced.py

Enhanced cleaning + mapping verification (HITL) for the RTGS pipeline.

Features:
- Loads CSV -> canonicalize column names
- Auto-detects bookings/delivered/date/year/month columns from heuristics
- Fills canonical columns: date, year, month, noofbookings, delivered
- Writes mapping profile, cleaned CSV, and a DQ report
- Calls llm_adapter.call_llm(...) to verify inferred mapping (HITL). Use --auto-approve
  to accept LLM suggestions automatically (helpful for demo runs).
Usage:
  python src/clean_enhanced.py --in data/cleaned/tankers_raw_merged.csv --out data/cleaned/tankers_cleaned_enhanced.csv [--auto-approve]
"""
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import pandas as pd

# local llm adapter (Gemini-only adapter in this repo)
try:
    from src import llm_adapter as llm_adapter_pkg  # when executed as module
except Exception:
    import llm_adapter as llm_adapter_pkg        # when executed directly

TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def infer_and_fix_dates_and_counts(df: pd.DataFrame, filename_hint_col: str = "_source_file") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    import calendar
    diag = {"found_cols": list(df.columns), "mapped": {}, "warnings": [], "sample_vals": {}}
    for c in df.columns:
        nonnull = df[c].dropna()
        diag["sample_vals"][c] = str(nonnull.iloc[0]) if len(nonnull) > 0 else None

    col_lower_map = {c.lower().strip(): c for c in df.columns}

    def pick(*patterns):
        # pick first column whose lowered name contains any of the patterns
        for p in patterns:
            for kl, orig in col_lower_map.items():
                if p in kl:
                    return orig
        return None

    # bookings: try many variants, also totservices/billdservices as fallback
    bookings_col = pick('noofbookings','no_of_bookings','number_of_bookings','bookings','booked','requests','request','totservices','billdservices')
    if bookings_col:
        diag['mapped']['noofbookings'] = bookings_col
        df[bookings_col] = pd.to_numeric(df[bookings_col], errors='coerce')
        df['noofbookings'] = df.get('noofbookings', df[bookings_col])
    else:
        diag['warnings'].append("no bookings-like column found")

    # delivered
    delivered_col = pick('delivered','delv','supplied','fulfilled','supply')
    if delivered_col:
        diag['mapped']['delivered'] = delivered_col
        df[delivered_col] = pd.to_numeric(df[delivered_col], errors='coerce')
        df['delivered'] = df.get('delivered', df[delivered_col])
    else:
        diag['warnings'].append("no delivered-like column found")

    # date/year/month: multiple patterns & month-name parsing
    date_col = pick('date','booking_date','request_date','req_date','created_at')
    if date_col:
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        diag['mapped']['date_from'] = date_col

    # year/month columns that could be strings like 'March' or 'Mar'
    year_col = pick('year')
    month_col = pick('month','mon')
    if month_col and month_col in df.columns:
        # attempt to parse textual months to numbers
        def month_to_num(x):
            if pd.isna(x): return None
            s = str(x).strip()
            if s.isdigit():
                return int(s)
            # try month name
            try:
                return list(calendar.month_abbr).index(s[:3].title())
            except Exception:
                try:
                    return list(calendar.month_name).index(s.title())
                except Exception:
                    return None
        df[month_col + "_num"] = df[month_col].apply(month_to_num)
        if df[month_col + "_num"].notna().any():
            df['month'] = df.get('month', df[month_col + "_num"])
            diag['mapped']['month_from_text'] = month_col

    # if year+month present combine into date
    if year_col and month_col and year_col in df.columns and month_col in df.columns:
        try:
            y = pd.to_numeric(df[year_col], errors='coerce')
            m = pd.to_numeric(df[month_col], errors='coerce').fillna(df.get(month_col + "_num"))
            mask = y.notna() & m.notna()
            if mask.any():
                df.loc[mask, 'date'] = pd.to_datetime(y.fillna(0).astype(int).astype(str) + "-" + m.astype(int).astype(str).str.zfill(2) + "-01", errors='coerce')
                diag['mapped']['date_from_year_month'] = (year_col, month_col)
        except Exception:
            pass

    # fallback from filename patterns in _source_file
    if (('date' not in df.columns) or df['date'].isna().all()) and filename_hint_col in df.columns:
        def try_from_filename(s):
            if not isinstance(s, str): return (None, None)
            # handle multiple formats, look for like 2024_3 or 2024-03 or 202403
            m = re.search(r'(20\d{2})[^\d]?(0?[1-9]|1[0-2])', s)
            if m:
                return int(m.group(1)), int(m.group(2))
            m2 = re.search(r'(0?[1-9]|1[0-2])[^\d]?(20\d{2})', s)
            if m2:
                return int(m2.group(2)), int(m2.group(1))
            return (None, None)
        ym = df[filename_hint_col].apply(lambda x: try_from_filename(str(x)))
        df['__src_y'] = ym.apply(lambda t: t[0])
        df['__src_m'] = ym.apply(lambda t: t[1])
        mask2 = df['__src_y'].notna() & df['__src_m'].notna()
        if mask2.any():
            df.loc[mask2, 'date'] = pd.to_datetime(df.loc[mask2, '__src_y'].astype(int).astype(str) + "-" + df.loc[mask2, '__src_m'].astype(int).astype(str).str.zfill(2) + "-01", errors='coerce')
            diag['mapped']['date_from_source_file'] = True

    # helpers
    if 'date' in df.columns:
        df['month_iso'] = df['date'].dt.strftime('%Y-%m')
        df['year'] = df.get('year', df['date'].dt.year)
        df['month'] = df.get('month', df['date'].dt.month)

    # coerce numeric columns
    for c in ['noofbookings', 'delivered', 'year', 'month']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # drop temp helpers
    for h in ['__src_y','__src_m'] + [col for col in df.columns if col.endswith("_num")]:
        if h in df.columns:
            df.drop(columns=[h], inplace=True)

    diag['counts'] = {'rows': len(df), 'date_nonnull': int(df.get('date').notna().sum() if 'date' in df else 0),
                      'bookings_nonnull': int(df.get('noofbookings').notna().sum() if 'noofbookings' in df else 0),
                      'delivered_nonnull': int(df.get('delivered').notna().sum() if 'delivered' in df else 0)}
    return df, diag


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def call_mapping_verification_llm(mapping: Dict[str, Any], sample: Dict[str, Any], auto_approve: bool = False) -> Tuple[bool, Dict]:
    """
    Ask LLM to verify suggested mapping. Returns (approved_bool, llm_result).
    llm_result: {"ok": bool, "suggestions": [ ... ] }
    """
    # Build prompt
    lines = [
        "You are a schema-mapping assistant. We have raw CSV columns and a suggested canonical mapping.",
        "Reply in strict JSON: {\"ok\": bool, \"suggestions\": [{\"raw\": <raw_col>, \"suggested\": <canonical>, \"reason\": <string>}, ...]}",
        "",
        "Raw columns (name -> sample value):"
    ]
    for k, v in sample.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Suggested canonical mapping:")
    for k, v in mapping.items():
        lines.append(f"- canonical '{k}' <- raw '{v}'")
    lines.append("")
    lines.append("If mapping looks correct, set ok=true and suggestions=[]. If some raw columns should map to *different* canonical names, list them in suggestions with reasons.")
    prompt = "\n".join(lines)

    try:
        llm_out = llm_adapter_pkg.call_llm(prompt=prompt, model=None, temperature=0.0, max_tokens=512)
    except Exception as e:
        # if llm call fails, return not approved and raw text
        return False, {"ok": False, "error": str(e), "raw_text": str(e)}

    # Try to parse JSON out of llm_out
    parsed = None
    try:
        parsed = json.loads(llm_out.strip())
    except Exception:
        # LLM might return code block or text; try to extract JSON snippet
        m = re.search(r'(\{.*\})', llm_out, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                parsed = {"ok": False, "raw_text": llm_out}
        else:
            parsed = {"ok": False, "raw_text": llm_out}

    if auto_approve:
        # For demo runs allow auto-approve but still return parsed result
        return True, parsed

    # return parsed approval status
    return bool(parsed.get("ok")), parsed


def main(infile: str, out: str, auto_approve: bool = False):
    pin = Path(infile)
    pout = Path(out)
    if not pin.exists():
        raise SystemExit(f"Input file not found: {pin}")

    print("[clean_enhanced] loading", infile)
    df = pd.read_csv(pin)

    df_fixed, diag = infer_and_fix_dates_and_counts(df)

    # prepare mapping profile for audit
    mapping = {
        "date": diag.get("mapped", {}).get("date_from") or diag.get("mapped", {}).get("date_from_year_month") or ("_source_file" if diag.get("mapped", {}).get("date_from_source_file") else None),
        "noofbookings": diag.get("mapped", {}).get("noofbookings") or None,
        "delivered": diag.get("mapped", {}).get("delivered") or None
    }

    # sample row values for LLM prompt
    sample_row = {}
    for c in df_fixed.columns:
        val = df_fixed[c].dropna().astype(str).head(1)
        sample_row[c] = val.iloc[0] if len(val) > 0 else None

    # Call LLM for mapping verification (HITL)
    print("Wrote mapping to outputs/profiles/mapping-{}.json".format(TS()))
    mapping_profile_path = Path("outputs/profiles") / f"mapping-{TS()}.json"
    write_json({"mapping": mapping, "diag": diag}, mapping_profile_path)

    print("[clean_enhanced] running mapping verification (HITL)...")
    approved, llm_result = call_mapping_verification_llm(mapping=mapping, sample=sample_row, auto_approve=auto_approve)

    # If not approved and not auto-approved, ask user
    if not approved and not auto_approve:
        print("=== LLM OUTPUT ===")
        print(json.dumps(llm_result, indent=2, ensure_ascii=False))
        ans = input("Approve mapping? [y=accept, e=edit mapping file, n=abort]\n> ").strip().lower()
        if ans == "e":
            print("Opening mapping file for editing:", mapping_profile_path)
            print("Edit the file then re-run the script.")
            raise SystemExit("Mapping edit requested. Re-run after editing.")
        if ans != "y":
            raise SystemExit("Mapping not approved. Aborting.")

    print("[clean_enhanced] mapping approved. Done.")

    # final cleanup and outputs
    pout.parent.mkdir(parents=True, exist_ok=True)
    df_fixed.to_csv(pout, index=False)
    print(f"Wrote cleaned CSV to {pout} ({len(df_fixed)} rows)")

    # write simple DQ report
    dq = {
        "ts": TS(),
        "rows": len(df_fixed),
        "counts": diag.get("counts", {}),
        "warnings": diag.get("warnings", []),
        "mapped": diag.get("mapped", {})
    }
    dq_path = Path("outputs") / f"dq_report-{TS()}.json"
    write_json(dq, dq_path)
    print("Wrote DQ report to", dq_path)

    # also write a short human-readable profile
    profile = {
        "ts": TS(),
        "mapping_profile": mapping,
        "counts": diag.get("counts", {}),
        "notes": diag.get("warnings", [])
    }
    prof_path = Path("outputs/profiles") / f"profile-{TS()}.json"
    write_json(profile, prof_path)
    print("Wrote profile to", prof_path)

    return pout, dq_path, prof_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--auto-approve", action="store_true", help="Auto-approve LLM mapping suggestions (useful for demo runs)")
    args = p.parse_args()
    main(args.infile, args.out, auto_approve=args.auto_approve)
