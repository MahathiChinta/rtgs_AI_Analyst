"""
Build a simple RAG index (passages.jsonl + optional embeddings.npy)
that is robustly data-agnostic: it auto-detects useful columns and
ensures numeric booking/delivery fields are present in the passage text.

Usage:
  python src/rag_builder_fallback.py --in data/cleaned/tankers_cleaned_enhanced.csv --out_dir outputs/rag --model all-MiniLM-L6-v2
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import re

# Try import sentence-transformers (optional)
HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

def detect_key_columns(df: pd.DataFrame):
    # preferred substrings in order (we search for these substrings in column names)
    prefer = ["year", "month", "date", "noofbookings", "no_of_bookings", "number_of_bookings",
              "bookings", "booked", "requests", "totservices", "billdservices",
              "delivered", "deliv", "supplied", "fulfilled", "supply",
              "section", "division", "area"]
    lowered = {c.lower(): c for c in df.columns}
    mapped = {}

    # substring search
    for p in prefer:
        for lowname, orig in lowered.items():
            if p in lowname and p not in mapped:
                # choose first matching original column for this preferred category
                # store by a canonical key (bookings -> noofbookings etc)
                if any(x in p for x in ("noofbookings","no_of_bookings","number_of_bookings","bookings","booked","requests","totservices","billdservices")):
                    if "noofbookings" not in mapped:
                        mapped["noofbookings"] = orig
                elif any(x in p for x in ("delivered","deliv","supplied","fulfilled","supply")):
                    if "delivered" not in mapped:
                        mapped["delivered"] = orig
                elif p in ("date","year","month"):
                    # map individually
                    if p == "date":
                        mapped["date"] = orig
                    if p == "year":
                        mapped["year"] = orig
                    if p == "month":
                        mapped["month"] = orig
                else:
                    # section/division/area
                    if p in ("section","division","area"):
                        mapped[p] = orig

    # fallback: choose top numeric columns as bookings/delivered if not found
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not any(k in mapped for k in ("noofbookings",)):
        # prefer numeric columns with obvious names
        fallback = None
        for c in numeric_cols:
            if any(x in c.lower() for x in ("count","tot","num","noof","book")):
                fallback = c; break
        if fallback is None and numeric_cols:
            fallback = numeric_cols[0]
        if fallback:
            mapped["noofbookings"] = fallback
    if not any(k in mapped for k in ("delivered",)):
        # pick a second numeric col if available
        if len(numeric_cols) >= 2:
            booked_col = mapped.get("noofbookings")
            for c in numeric_cols:
                if c != booked_col:
                    mapped["delivered"] = c
                    break

    # ensure mapped values are actual column names
    final = {}
    for k, v in mapped.items():
        if v in df.columns:
            final[k] = v
    # include best-effort canonical names even if not all present
    return final

def make_text_for_row(row, mapped_cols):
    pieces = []
    # canonical fields
    for k in ("year","month","date"):
        col = mapped_cols.get(k)
        if col and pd.notna(row.get(col)):
            pieces.append(f"{k}: {row.get(col)}")
    # location fields
    for k in ("division","section","area"):
        col = mapped_cols.get(k)
        if col and pd.notna(row.get(col)):
            pieces.append(f"{k}: {row.get(col)}")
    # booking/delivery
    for k,out_label in (("noofbookings","booked"), ("delivered","delivered")):
        col = mapped_cols.get(k)
        if col:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() != "":
                pieces.append(f"{out_label}: {val}")
    # fallback: include _source_file or any id
    if "_source_file" in row and pd.notna(row["_source_file"]):
        pieces.append(f"source: {row['_source_file']}")
    return " | ".join(pieces) if pieces else None

def build_passages(df: pd.DataFrame, out_dir: Path):
    mapped = detect_key_columns(df)
    # generate passages with an informative text field and numeric meta
    passages = []
    for idx, row in df.reset_index(drop=True).iterrows():
        txt = make_text_for_row(row, mapped)
        if not txt:
            # create a generic row summary using first few columns
            mini = ", ".join([f"{c}:{row.get(c)}" for c in df.columns[:6]])
            txt = mini
        # build meta: include mapped column names and extracted numeric values if possible
        meta = {"mapped": mapped, "_source_row": int(idx)}
        # try to extract numeric booking/delivered as floats
        for k in ("noofbookings", "delivered"):
            col = mapped.get(k)
            if col and col in df.columns:
                try:
                    v = row.get(col)
                    if pd.isna(v) or str(v).strip() == "":
                        meta[k] = None
                    else:
                        # try numeric conversion
                        num = float(re.sub(r'[^\d.-]+','', str(v))) if re.search(r'\d', str(v)) else None
                        meta[k] = num
                except Exception:
                    meta[k] = None
        # include normalized date string if possible
        if mapped.get("date") and mapped["date"] in df.columns:
            try:
                d = pd.to_datetime(row.get(mapped["date"]), errors="coerce")
                meta["date_iso"] = d.strftime("%Y-%m-%d") if not pd.isna(d) else None
            except Exception:
                meta["date_iso"] = None
        else:
            # try year/month
            ycol = mapped.get("year"); mcol = mapped.get("month")
            if ycol in df.columns or mcol in df.columns:
                try:
                    y = row.get(ycol) if ycol in df.columns else None
                    m = row.get(mcol) if mcol in df.columns else None
                    if y and m:
                        meta["date_iso"] = f"{int(float(y))}-{int(float(m)):02d}-01"
                except Exception:
                    pass

        passage = {
            "id": int(idx),
            "text": str(txt),
            "meta": meta
        }
        passages.append(passage)
    # write jsonl
    out_dir.mkdir(parents=True, exist_ok=True)
    pjsonl = out_dir / "passages.jsonl"
    with open(pjsonl, "w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Wrote passages.jsonl (count {})".format(len(passages)))
    return pjsonl, mapped

def build_embeddings(passages_jsonl: Path, out_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
    if not HAS_ST:
        print("sentence-transformers not available; skipping embeddings build.")
        return None
    texts = []
    with open(passages_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            texts.append(j.get("text",""))
    print("Computing embeddings with", model_name)
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True)
    np.save(out_dir / "embeddings.npy", emb)
    print("Wrote embeddings.npy")
    return out_dir / "embeddings.npy"

def main(infile, out_dir, embed_model):
    df = pd.read_csv(infile, dtype=str)
    outd = Path(out_dir)
    pjsonl, mapped = build_passages(df, outd)
    if embed_model and HAS_ST:
        build_embeddings(pjsonl, outd, model_name=embed_model)
    meta = {"count": len(df), "mapped_columns": mapped}
    with open(outd / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("Wrote meta.json")
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out_dir", dest="out_dir", required=True)
    p.add_argument("--model", dest="model", default="all-MiniLM-L6-v2")
    args = p.parse_args()
    main(args.infile, args.out_dir, args.model)
