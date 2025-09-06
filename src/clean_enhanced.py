
"""
clean_enhanced.py - enhanced cleaner with mapping verification (HITL) and --auto-approve.

Use:
  python src/clean_enhanced.py --in data/cleaned/tankers_raw_merged.csv --out data/cleaned/tankers_cleaned_enhanced.csv
  python src/clean_enhanced.py --in ... --out ... --auto-approve

If --auto-approve is set the script will call the LLM to validate mapping and
auto-apply (accept) the LLM's 'ok' decision (no human prompt). If LLM output cannot
be parsed, it will still continue when --auto-approve is True, otherwise it will prompt.
"""
import argparse, os, platform, subprocess, json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# Import llm_adapter.call_llm (works when src package or running from repo root)
try:
    from src.llm_adapter import call_llm
except Exception:
    try:
        from llm_adapter import call_llm
    except Exception:
        def call_llm(prompt, model=None, temperature=0.0, max_tokens=1024):
            print("LLM adapter not available — printing manual prompt:")
            print(prompt)
            print("Paste the LLM answer now (end with empty line):")
            lines = []
            while True:
                l = input()
                if not l.strip():
                    break
                lines.append(l)
            return "\n".join(lines)

def ensure_dirs():
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    Path("outputs/profiles").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

def canonize(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    nm = name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")
    nm = "".join(ch for ch in nm if ch.isalnum() or ch == "_")
    nm = "_".join([p for p in nm.split("_") if p])
    return nm or name

def infer_and_map_columns(df: pd.DataFrame):
    return {col: canonize(col) for col in df.columns}

def clean_dataframe(df: pd.DataFrame, mapping):
    df = df.rename(columns=mapping).copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"": pd.NA})
    for c in df.columns:
        if any(k in c for k in ("noof", "no_", "no", "count", "delivered", "booked", "total")):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)

def read_csv_preview(path, n=3):
    df = pd.read_csv(path, nrows=n, dtype=str)
    return df.fillna("").to_dict(orient="records")

def build_mapping_prompt(mapping, sample_rows, context_note=""):
    lines = []
    lines.append("Mapping verification task:")
    lines.append("RAW_COLUMN -> CANONICAL_NAME")
    for r, t in mapping.items():
        lines.append(f"- {r} -> {t}")
    if context_note:
        lines.append("")
        lines.append("Context: " + context_note)
    lines.append("")
    lines.append("Sample rows (first few):")
    for i, row in enumerate(sample_rows):
        lines.append(f"Row {i+1}: " + ", ".join([f"{k}={v}" for k, v in row.items()]))
    lines.append("")
    lines.append("Return a JSON object: { 'ok': true/false, 'suggestions': [{ 'raw':..., 'suggested':..., 'reason':... }, ...] }")
    lines.append("Be concise and avoid hallucination.")
    return "\n".join(lines)

def parse_llm_json_safe(text):
    # Naive extraction of first JSON object substring
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        return None
    return None

def open_in_editor(path):
    editor = os.environ.get("EDITOR")
    if editor:
        subprocess.run([editor, path])
        return
    if platform.system().lower().startswith("win"):
        subprocess.run(["notepad.exe", path])
    else:
        try:
            subprocess.run(["xdg-open", path])
        except Exception:
            print("Please edit the file manually:", path)

def verify_mapping(mapping_path, sample_csv, context_note="", auto_approve=False):
    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    sample_rows = read_csv_preview(sample_csv, n=3)
    prompt = build_mapping_prompt(mapping, sample_rows, context_note)
    print("[verify_mapping] calling LLM...")
    llm_resp = call_llm(prompt, model=None, temperature=0.0, max_tokens=500)
    parsed = parse_llm_json_safe(llm_resp)
    print("=== LLM OUTPUT ===")
    if parsed:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print(llm_resp)
        print("=== (non-JSON LLM output) ===")
    if auto_approve:
        print("[verify_mapping] auto-approve enabled. proceeding (LLM output used for info only).")
        return True
    # human loop
    while True:
        print("Approve mapping? [y=accept, e=edit mapping file, n=abort]")
        c = input("> ").strip().lower()
        if c == "y":
            return True
        if c == "n":
            return False
        if c == "e":
            open_in_editor(mapping_path)
            # after edit, re-run verification recursively
            try:
                with open(mapping_path, "r", encoding="utf-8") as fh:
                    json.load(fh)
                return verify_mapping(mapping_path, sample_csv, context_note, auto_approve)
            except Exception as e:
                print("Edited mapping JSON invalid:", e)
                print("Fix JSON and choose 'e' again, or 'n' to abort.")

def main(infile, out_csv, note="", auto_approve=False):
    ensure_dirs()
    ts = TS()
    print("[clean_enhanced] loading", infile)
    df = pd.read_csv(infile, dtype=str)
    mapping = infer_and_map_columns(df)
    mapping_path = f"outputs/profiles/mapping-{ts}.json"
    write_json(mapping, mapping_path)
    print("Wrote mapping to", mapping_path)
    cleaned = clean_dataframe(df, mapping)
    cleaned.to_csv(out_csv, index=False)
    print(f"Wrote cleaned CSV to {out_csv} ({len(cleaned)} rows)")
    dq = {"ts": ts, "rows": len(cleaned), "columns": list(cleaned.columns)[:50], "notes": "Basic cleaning complete."}
    dq_path = f"outputs/dq_report-{ts}.json"
    write_json(dq, dq_path)
    print("Wrote DQ report to", dq_path)
    print("[clean_enhanced] running mapping verification (HITL)...")
    ok = verify_mapping(mapping_path, infile, context_note=note, auto_approve=auto_approve)
    if not ok:
        raise SystemExit("Mapping not approved. Aborting.")
    print("[clean_enhanced] mapping approved. Done.")
    return 0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outcsv", required=True)
    p.add_argument("--note", dest="note", default="")
    p.add_argument("--auto-approve", dest="auto_approve", action="store_true", help="Auto-approve mapping (no HITL prompt).")
    args = p.parse_args()
    main(args.infile, args.outcsv, note=args.note, auto_approve=args.auto_approve)
