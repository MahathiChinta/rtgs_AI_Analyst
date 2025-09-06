"""
Build a FAISS index (persisted) from a cleaned CSV to speed up semantic QA.
Uses sentence-transformers to embed and faiss for indexing.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

def build_faiss(infile, out_dir, model_name="all-MiniLM-L6-v2"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    # create simple passages (string per row)
    passages = (df.fillna("").astype(str).apply(lambda r: " | ".join([f"{c}:{r[c]}" for c in df.columns]), axis=1)).tolist()

    # local import to avoid heavy deps if not used
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss

    model = SentenceTransformer(model_name)
    embeddings = model.encode(passages, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # dot-product (works when embeddings are normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # persist index and metadata
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "passages.jsonl", "w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps({"text": p}) + "\n")
    print("Wrote FAISS index to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()
    build_faiss(args.infile, args.out_dir, args.model)
