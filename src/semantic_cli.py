#!/usr/bin/env python3
"""
src/semantic_cli.py

Interactive CLI for semantic Q&A using existing RAG artifacts.

Usage examples:
  # interactive prompt (type question at prompt)
  python src/semantic_cli.py --rag_dir outputs/rag --k 6 --model gemini-2.5-pro --temp 0.0

  # one-shot question
  python src/semantic_cli.py --rag_dir outputs/rag --k 6 --model gemini-2.5-pro --temp 0.0 --question "Which localities show the largest unmet tanker demand?"

Notes:
- Expects `passages.jsonl` in --rag_dir. If `embeddings.npy` exists, uses sentence-transformers for semantic search.
- Falls back to token-overlap scoring if embeddings or sentence-transformers package are absent.
- Saves evidence and report in outputs/ and a trace in traces/.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRACES = ROOT / "traces"

# local adapter that your pipeline already uses
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import llm_adapter
 # uses call_llm(prompt, model, temperature, max_tokens)

# optional local imports (for embeddings)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False

def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def load_passages(rag_dir: str) -> List[Dict]:
    p = Path(rag_dir)
    pj = p / "passages.jsonl"
    if not pj.exists():
        raise FileNotFoundError(f"passages.jsonl not found in {rag_dir}")
    passages = []
    with open(pj, "r", encoding="utf-8") as fh:
        for line in fh:
            passages.append(json.loads(line))
    # ensure deterministic order stable by id
    passages.sort(key=lambda x: str(x.get("id", "")))
    return passages

def load_embeddings_if_present(rag_dir: str):
    emb_path = Path(rag_dir) / "embeddings.npy"
    if emb_path.exists() and HAS_EMBED:
        emb = np.load(emb_path)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return emb, model
    return None, None

def score_passages_with_embedding(query: str, passages: List[Dict], emb_matrix, embed_model) -> List[Tuple[int, float]]:
    import numpy as np
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    # normalize
    def norm(v): return v / (np.linalg.norm(v) + 1e-12)
    qn = norm(q_emb)
    # compute cosine similarities
    scores = []
    for i, v in enumerate(emb_matrix):
        scores.append((i, float(np.dot(qn, norm(v)))))
    # sort desc by score, stable by index for tie-break
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores

def score_passages_token_overlap(query: str, passages: List[Dict]) -> List[Tuple[int, float]]:
    q_tokens = set([t.lower() for t in query.split() if len(t) > 2])
    scores = []
    for i, p in enumerate(passages):
        text = p.get("text", "")
        p_tokens = set([t.lower() for t in text.split() if len(t) > 2])
        inter = q_tokens.intersection(p_tokens)
        score = len(inter) / (len(p_tokens) + 1e-6)
        scores.append((i, float(score)))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores

def build_prompt(evidence_items: List[str], question: str) -> str:
    header = "You are an evidence-focused analyst. Use ONLY the provided evidence (quote E1..En). If evidence insufficient, say 'Insufficient evidence' and recommend a data collection step.\n\n"
    evidence_text = "\n".join([f"[E{idx+1}] {ev}" for idx, ev in enumerate(evidence_items)])
    instructions = (
        "\nInstructions:\n"
        "- Answer only from the Evidence above.\n"
        "- Provide: (A) one-line summary; (B) up to 3 concise policy recommendations (one sentence each).\n"
        "- If evidence insufficient, respond 'Insufficient evidence' and suggest one concrete data collection improvement.\n"
        "- Be concise (max 150 words)."
    )
    prompt = f"{header}Question:\n{question}\n\nEvidence:\n{evidence_text}\n\n{instructions}\n"
    return prompt

def save_outputs(evidence: List[Dict], report_text: str, question: str):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    t = ts()
    ev_path = OUTPUTS / f"evidence-{t}.json"
    rpt_path = OUTPUTS / f"report-{t}.md"
    trace_path = TRACES / f"run-{t}.json"
    with open(ev_path, "w", encoding="utf-8") as fh:
        json.dump({"question": question, "evidence": evidence}, fh, indent=2)
    with open(rpt_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"ts": t, "question": question, "evidence_count": len(evidence)}, fh, indent=2)
    return ev_path, rpt_path, trace_path

def interactive_loop(args):
    passages = load_passages(args.rag_dir)
    emb_matrix, embed_model = load_embeddings_if_present(args.rag_dir)

    print("Interactive semantic CLI. Type your question, or 'quit' to exit.")
    while True:
        try:
            q = args.question or input("\nQuestion> ").strip()
            # if question given as arg then run once and exit
            if not q:
                continue
            if q.lower() in ("quit", "exit"):
                break

            # retrieval
            if emb_matrix is not None and embed_model is not None:
                scores = score_passages_with_embedding(q, passages, emb_matrix, embed_model)
            else:
                scores = score_passages_token_overlap(q, passages)

            k = int(args.k)
            top_k = scores[:k]
            evidence_items = []
            evidence_meta = []
            for idx, score in top_k:
                p = passages[idx]
                evidence_items.append(p.get("text", ""))
                evidence_meta.append({"id": p.get("id"), "score": score, "meta": p.get("meta", {})})

            prompt = build_prompt(evidence_items, q)
            # call LLM via your llm_adapter (Gemini-only adapter)
            model = args.model
            temp = float(args.temp)
            print("\n=== Sending prompt to LLM (this may be slow) ===\n")
            try:
                llm_out = llm_adapter.call_llm(prompt=prompt, model=model, temperature=temp, max_tokens=int(args.max_tokens))
            except Exception as e:
                print("LLM adapter call failed:", repr(e))
                llm_out = "NO_AUTOMATIC_LLM_RESPONSE — prompt printed; paste model answer here."
                print(prompt)

            # print and save outputs
            print("\n=== LLM Answer ===\n")
            print(llm_out)
            ev_path, rpt_path, trace_path = save_outputs(evidence_meta, llm_out, q)
            print(f"\nSaved evidence -> {ev_path}\nReport -> {rpt_path}\nTrace -> {trace_path}")

            # if question provided via CLI arg, run one-shot and exit
            if args.question:
                break

        except KeyboardInterrupt:
            print("\nExiting.")
            break

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag_dir", default=str(OUTPUTS / "rag"), help="RAG directory containing passages.jsonl and optional embeddings.npy")
    p.add_argument("--k", default=6, type=int, help="Number of evidence passages to retrieve")
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"), help="LLM model to use (Gemini)")
    p.add_argument("--temp", default=os.getenv("DEFAULT_TEMPERATURE", "0.0"), help="Sampling temperature")
    p.add_argument("--max_tokens", default=512, help="Max tokens for LLM output")
    p.add_argument("--question", default=None, help="If provided, the CLI will run one-shot and exit")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    interactive_loop(args)
