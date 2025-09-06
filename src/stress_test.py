import argparse
import pandas as pd
import json
from datetime import datetime

def validate_csv(path: str):
    df = pd.read_csv(path)
    out = {
        "file": str(path),
        "rows": int(len(df)),
        "cols": [str(c) for c in df.columns],
        "checks": []
    }

    # Non-empty check
    out["checks"].append({
        "name": "non_empty",
        "ok": bool(len(df) > 0),
        "detail": f"rows={int(len(df))}"
    })

    # Missingness
    for c in df.columns:
        pct = float(df[c].isna().mean() * 100)
        out["checks"].append({
            "name": "missing_pct",
            "col": str(c),
            "pct_missing": round(pct, 2),
            "ok": bool(pct < 40.0)
        })

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", required=True)
    parser.add_argument("--out", dest="out_json", default="outputs/validation.json")
    args = parser.parse_args()

    res = validate_csv(args.input_csv)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_json.replace(".json", f"-{ts}.json")

    # Force convert everything to safe JSON types
    def force_safe(obj):
        if isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [force_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): force_safe(v) for k, v in obj.items()}
        return str(obj)

    safe_res = force_safe(res)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe_res, f, indent=2, ensure_ascii=False)

    print(f"Wrote validation results to {out_path}")

if __name__ == "__main__":
    main()
