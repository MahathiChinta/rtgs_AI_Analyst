"""
Build a local RAG index (FAISS) from cleaned CSV.
Outputs:
  outputs/rag/index.faiss
  outputs/rag/metadata.json
  outputs/rag/model_name.txt

This file is deterministic and runs without an LLM API key.
"""
import argparse, json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

def make_text(row):
    # create a compact evidence passage for a row
    parts = []
    parts.append(f"row_id:{row.get('row_id','')}")
    parts.append(f"loc:{row.get('booking_location','')}")
    parts.append(f"month:{row.get('month_start','')}")
    parts.append(f"booked:{row.get('tankers_booked','')}")
    parts.append(f"delivered:{row.get('tankers_delivered','')}")
    parts.append(f"gap:{row.get('delivery_gap','')}")
    return " | ".join(parts)

def main(infile, out_dir, model_name="all-MiniLM-L6-v2"):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile, dtype=str, keep_default_na=False, na_values=["","NA","N/A",None])
    if 'row_id' not in df.columns:
        df = df.reset_index().rename(columns={'index':'row_id'})
        df['row_id'] = df['row_id'].apply(lambda x: f"R{x+1}")

    docs = [make_text(r) for _, r in df.iterrows()]

    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)

    print("Computing embeddings...")
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    print("Embedding dim:", dim)

    index = faiss.IndexFlatIP(dim)  # use inner product on normalized vectors -> cosine
    # normalize embeddings
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, str(outdir / "index.faiss"))

    metadata = []
    for i, (_, row) in enumerate(df.iterrows()):
        metadata.append({
            "id": i,
            "row_id": row.get('row_id'),
            "source_file": row.get('_source_file',''),
            "booking_location": row.get('booking_location',''),
            "month_start": row.get('month_start',''),
            "tankers_booked": row.get('tankers_booked',''),
            "tankers_delivered": row.get('tankers_delivered',''),
            "delivery_gap": row.get('delivery_gap',''),
            "text": docs[i]
        })
    with open(outdir / "metadata.json", "w", encoding="utf8") as fh:
        json.dump(metadata, fh, indent=2)
    with open(outdir / "model_name.txt", "w", encoding="utf8") as fh:
        fh.write(model_name)
    print("Saved FAISS index + metadata to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out_dir", dest="out_dir", default="outputs/rag")
    parser.add_argument("--model", dest="model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args.infile, args.out_dir, args.model)
