#!/usr/bin/env python3
"""
semantic_qa.py - run evidence retrieval + LLM answer; save artifacts.
Usage:
python src/semantic_qa.py --q "Your question" --rag_dir outputs/rag --k 6
"""
import argparse, json, textwrap
from pathlib import Path
from datetime import datetime
from retriever import Retriever
from llm_adapter import call_llm

SYSTEM_INSTR = "You are an evidence-focused analyst. Use ONLY the provided evidence (quote E1..En). If evidence insufficient, say 'Insufficient evidence' and recommend a data collection step."

def build_prompt(question, hits):
    ev = []
    for i, h in enumerate(hits, start=1):
        text = h.get("text","")
        ev.append(f"[E{i}] {text}")
    ev_block = "\n".join(ev)
    prompt = f"""{SYSTEM_INSTR}

Question:
{question}

Evidence:
{ev_block}

Instructions:
- Answer only from the Evidence above.
- Provide: (A) one-line summary; (B) up to 3 concise policy recommendations (one sentence each).
- If evidence insufficient, respond 'Insufficient evidence' and suggest one concrete data collection improvement.
- Be concise (max 150 words).
"""
    return prompt

def ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_artifacts(question, hits, answer, out_dir="outputs"):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    evid_path = outdir / f"evidence-{ts()}.json"
    rep_path = outdir / f"report-{ts()}.md"
    with open(evid_path, "w", encoding="utf8") as fh:
        json.dump({"question": question, "hits": hits}, fh, indent=2, ensure_ascii=False)
    with open(rep_path, "w", encoding="utf8") as fh:
        fh.write(f"# Semantic QA Report\n\n**Question:** {question}\n\n")
        fh.write("## Top Evidence\n\n")
        for i,h in enumerate(hits, start=1):
            fh.write(f"- E{i} (score={h.get('score'):.3f}): {h.get('text')}\n")
        fh.write("\n## Answer (LLM)\n\n")
        fh.write(answer if answer else "_No automatic LLM response — prompt printed for manual paste._")
    return str(evid_path), str(rep_path)

def main(question, rag_dir="outputs/rag", k=6, llm_provider=None, llm_model=None, temp=0.0):
    retriever = Retriever(rag_dir=rag_dir)
    hits = retriever.query(question, k=k)
    if not hits:
        print("No evidence found.")
        return
    prompt = build_prompt(question, hits)
    answer = call_llm(prompt, provider=llm_provider, model=llm_model, temperature=temp, max_tokens=300)
    evid_path, rep_path = save_artifacts(question, hits, answer)
    print("Evidence saved to:", evid_path)
    print("Report saved to:", rep_path)
    if answer:
        print("\n=== LLM Answer ===\n")
        print(answer)
    else:
        print("\nManual prompt printed — paste into any LLM UI to get the answer, then copy into the report file.")
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q", dest="question", required=True)
    p.add_argument("--rag_dir", default="outputs/rag")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--provider", default=None)
    p.add_argument("--llm_model", default=None)
    p.add_argument("--temp", type=float, default=0.0)
    args = p.parse_args()
    main(args.question, rag_dir=args.rag_dir, k=args.k, llm_provider=args.provider, llm_model=args.llm_model, temp=args.temp)
