"""
semantic_qa.py - run evidence retrieval + LLM answer; save artifacts into outputs/final/.
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

def save_artifacts(question, hits, answer, out_dir="outputs/final"):
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
            score = h.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            fh.write(f"- E{i} (score={score_str}): {h.get('text')}\n")
        fh.write("\n## Answer (LLM)\n\n")
        fh.write(answer or "_No automatic LLM response — prompt printed for manual paste._")
    return str(evid_path), str(rep_path)

def run_deterministic_fallback(cleaned_csv_path: str):
    try:
        from src.quick_analytics import deterministic_fallback_summary
    except Exception:
        try:
            from quick_analytics import deterministic_fallback_summary
        except Exception:
            deterministic_fallback_summary = None

    if deterministic_fallback_summary:
        try:
            return deterministic_fallback_summary(cleaned_csv_path=cleaned_csv_path)
        except Exception as e:
            return f"(fallback analytics failed: {e})"

    import pandas as pd
    try:
        df = pd.read_csv(cleaned_csv_path)
        booking_col = next((c for c in df.columns if "book" in c.lower()), None)
        delivered_col = next((c for c in df.columns if "deliv" in c.lower()), None)
        loc_col = df.columns[0] if len(df.columns) > 0 else None
        if booking_col and delivered_col and loc_col:
            df["_book"] = pd.to_numeric(df[booking_col], errors="coerce").fillna(0)
            df["_deliv"] = pd.to_numeric(df[delivered_col], errors="coerce").fillna(0)
            df["_gap"] = df["_book"] - df["_deliv"]
            top = df.groupby(loc_col)["_gap"].sum().nlargest(5)
            total_book = int(df["_book"].sum())
            total_deliv = int(df["_deliv"].sum())
            summary = [
                f"Total bookings: {total_book}",
                f"Total delivered: {total_deliv}",
                f"Total unmet (booked - delivered): {total_book - total_deliv}",
                "",
                "Top 5 locations by unmet quantity:",
                top.to_string()
            ]
            return "\n".join(summary)
        else:
            return "Deterministic fallback: could not auto-detect booking/delivered columns."
    except Exception as e:
        return f"Deterministic fallback error: {e}"

def main(question, rag_dir="outputs/rag", k=6, llm_provider=None, llm_model=None, temp=0.0):
    retriever = Retriever(rag_dir=rag_dir)
    hits = retriever.query(question, k=k)
    if not hits:
        print("No evidence found.")
        return
    prompt = build_prompt(question, hits)

    try:
        answer = call_llm(prompt, model=llm_model, temperature=temp, max_tokens=300)
    except TypeError:
        try:
            answer = call_llm(prompt, llm_model, temp, 300)
        except Exception as e:
            print("LLM adapter failed:", e)
            answer = None
    except Exception as e:
        print("LLM adapter failure:", e)
        answer = None

    evid_path, rep_path = save_artifacts(question, hits, answer or "")
    print("Evidence saved to:", evid_path)
    print("Report saved to:", rep_path)

    ans_norm = (answer or "").strip().lower()
    insufficient_triggers = ("insufficient evidence", "insufficient data", "evidence insufficient", "not enough evidence", "no evidence")
    if any(t in ans_norm for t in insufficient_triggers):
        print("\nLLM indicates insufficient evidence — running deterministic fallback analysis...\n")
        cleaned_guess = "data/cleaned/tankers_cleaned_enhanced.csv"
        fallback_text = run_deterministic_fallback(cleaned_guess)
        try:
            with open(rep_path, "a", encoding="utf8") as fh:
                fh.write("\n\n---\n\n### Deterministic fallback analysis\n\n")
                fh.write(fallback_text + "\n")
            print("Appended deterministic fallback to report:", rep_path)
        except Exception as e:
            print("Could not append deterministic fallback to report:", e)

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
