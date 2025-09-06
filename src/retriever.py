
"""
retriever.py - simple RAG retriever with fallbacks.
Usage:
from retriever import Retriever
r = Retriever(rag_dir="outputs/rag", model_name="all-MiniLM-L6-v2", use_faiss=False)
hits = r.query("Which localities show largest unmet demand?", k=6)
Each hit is {"id":..., "text":..., "score":..., "meta": {...}}
"""
import json, os, numpy as np
from pathlib import Path
from typing import List
from math import sqrt

class Retriever:
    def __init__(self, rag_dir="outputs/rag", model_name="all-MiniLM-L6-v2", use_faiss=False):
        self.rag_dir = Path(rag_dir)
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.passages = []
        self.embeddings = None
        self._load()

    def _load(self):
        p = self.rag_dir / "passages.jsonl"
        if p.exists():
            with open(p, "r", encoding="utf8") as fh:
                for line in fh:
                    self.passages.append(json.loads(line))
        # try embeddings.npy
        embp = self.rag_dir / "embeddings.npy"
        if embp.exists():
            try:
                self.embeddings = np.load(embp)
            except Exception:
                self.embeddings = None
        tfidf_p = self.rag_dir / "tfidf_matrix.npy"
        if tfidf_p.exists() and self.embeddings is None:
            try:
                self.embeddings = np.load(tfidf_p)
            except Exception:
                self.embeddings = None
        # if nothing else, passages list is available

    def _cosine(self, a, b):
        # a and b are 1D numpy arrays
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an == 0 or bn == 0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def _embed_query(self, text: str):
        # Try to embed with sentence-transformers if available and embeddings exist
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            vec = model.encode([text], show_progress_bar=False)[0]
            return vec
        except Exception:
            # Try TF-IDF vectorizer
            try:
                import pickle
                vec_path = self.rag_dir / "vectorizer.pkl"
                if vec_path.exists():
                    vec = pickle.load(open(vec_path, "rb"))
                    v = vec.transform([text]).toarray()[0]
                    return v
            except Exception:
                pass
        return None

    def query(self, question: str, k: int = 6) -> List[dict]:
        # If we have embeddings, embed query and compute cosine with stored matrix
        if self.embeddings is not None:
            qv = self._embed_query(question)
            if qv is None:
                # fallback to simple scoring by string overlap
                return self._substring_rank(question, k)
            scores = []
            for i, vec in enumerate(self.embeddings):
                sim = self._cosine(np.asarray(vec), np.asarray(qv))
                scores.append((i, float(sim)))
            scores.sort(key=lambda x: x[1], reverse=True)
            hits = []
            for idx, score in scores[:k]:
                p = self.passages[idx]
                hits.append({"id": p.get("id"), "text": p.get("text"), "score": float(score), "meta": p.get("meta", {})})
            return hits
        else:
            return self._substring_rank(question, k)

    def _substring_rank(self, question, k):
        q = question.lower()
        scored = []
        for i, p in enumerate(self.passages):
            text = p.get("text","").lower()
            # score by count of shared tokens
            qtok = set(q.split())
            ttok = set(text.split())
            overlap = len(qtok.intersection(ttok))
            scored.append((i, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for idx, score in scored[:k]:
            p = self.passages[idx]
            hits.append({"id": p.get("id"), "text": p.get("text"), "score": float(score), "meta": p.get("meta", {})})
        return hits
