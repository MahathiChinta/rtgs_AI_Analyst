
"""
rag_builder_fallback.py
Build a lightweight evidence store (outputs/rag):
- passages.jsonl (one JSON per line)
- embeddings.npy (if sentence-transformers available)
- vectorizer.pkl (if TF-IDF fallback used)
Usage:
python src/rag_builder_fallback.py --in data/cleaned/tankers_cleaned_enhanced.csv --out_dir outputs/rag --model all-MiniLM-L6-v2
"""
import argparse, json, os, pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

def build_passages_from_df(df, text_fields=None, max_len_chars=1000):
    passages = []
    for idx, row in df.iterrows():
        parts = []
        if text_fields:
            for f in text_fields:
                if f in row and pd.notna(row[f]):
                    parts.append(str(row[f]))
        else:
            # default: join small set of columns to form a short passage
            for c in df.columns[:6]:
                parts.append(f"{c}: {row.get(c,'')}")
        text = " | ".join([p for p in parts if p])
        text = text[:max_len_chars]
        passages.append({"id": str(idx), "text": text, "meta": {"row": int(idx)}})
    return passages

def try_sentence_transformers(model_name):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as e:
        return None

def try_tfidf():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer
    except Exception:
        return None

def main(infile, out_dir, model):
    df = pd.read_csv(infile, dtype=str)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    passages = build_passages_from_df(df)
    # write passages
    passages_path = out_dir / "passages.jsonl"
    with open(passages_path, "w", encoding="utf8") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    st = try_sentence_transformers(model)
    if st is not None:
        print("Using sentence-transformers for embeddings.")
        embedder = st
        texts = [p["text"] for p in passages]
        embeddings = embedder.encode(texts, show_progress_bar=False)
        emb_arr = np.array(embeddings, dtype=np.float32)
        np.save(out_dir / "embeddings.npy", emb_arr)
        meta = {"method": "sentence-transformers", "model": model, "n_passages": len(passages)}
        with open(out_dir / "meta.json", "w", encoding="utf8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print("Wrote embeddings.npy and passages.jsonl")
        return

    # TF-IDF fallback
    Tfidf = try_tfidf()
    if Tfidf is not None:
        print("Sentence-transformers not available; using TF-IDF fallback.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=5000, stop_words='english')
        texts = [p["text"] for p in passages]
        X = vec.fit_transform(texts)
        # save sparse matrix as dense if small, else save vectorizer + matrix slices
        np.save(out_dir / "tfidf_matrix.npy", X.toarray())
        with open(out_dir / "vectorizer.pkl", "wb") as fh:
            pickle.dump(vec, fh)
        meta = {"method": "tfidf", "n_passages": len(passages)}
        with open(out_dir / "meta.json", "w", encoding="utf8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print("Wrote tfidf_matrix.npy and vectorizer.pkl")
        return

    # Minimal fallback: no embeddings, keep passages only
    meta = {"method": "passages_only", "n_passages": len(passages)}
    with open(out_dir / "meta.json", "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("No embedding library available. Wrote passages.jsonl and meta.json (passages only).")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out_dir", dest="out_dir", required=True)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    args = p.parse_args()
    main(args.infile, args.out_dir, args.model)
