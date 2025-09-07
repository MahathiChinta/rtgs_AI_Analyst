"""
quick_analytics.py
Simple numeric summary + plot for cleaned CSVs (fast, zero-config).

Usage:
python src/quick_analytics.py --in data/cleaned/tankers_cleaned_enhanced.csv --groupby section --metric delivered --top 10

This script:
 - prints top N groups by metric sum
 - saves outputs/plots/<metric>_by_<groupby>.png
 - saves outputs/summary-<metric>-by-<groupby>.csv
"""
import argparse, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main(infile, groupby, metric, top, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(infile)
    print("Loaded:", infile)
    print("Columns:", list(df.columns))
    # try to find metric if not provided
    if metric is None:
        numerics = df.select_dtypes(include=['number']).columns.tolist()
        if not numerics:
            raise SystemExit("No numeric columns found. Provide --metric explicitly.")
        metric = numerics[0]
        print("Auto-selected metric:", metric)
    if groupby not in df.columns:
        raise SystemExit(f"Group-by column '{groupby}' not found in CSV.")
    df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
    summary = df.groupby(groupby)[metric].sum().sort_values(ascending=False)
    top_k = summary.head(top)
    print("\nTop", top, "groups by", metric)
    print(top_k.to_string())
    csv_out = out_dir / f"summary-{metric}-by-{groupby}.csv"
    top_k.reset_index().rename(columns={metric: f"{metric}_sum"}).to_csv(csv_out, index=False)
    print("Wrote summary CSV:", csv_out)
    png_out = out_dir / f"{metric}_by_{groupby}.png"
    plt.figure(figsize=(8, max(3, 0.4*len(top_k))))
    top_k.sort_values().plot.barh()
    plt.title(f"Top {top} {groupby} by {metric}")
    plt.xlabel(metric)
    plt.tight_layout()
    plt.savefig(png_out)
    plt.close()
    print("Wrote plot:", png_out)

# optional deterministic fallback function for semantic_qa/semantic_cli
def deterministic_fallback_summary(cleaned_csv_path: str):
    import pandas as pd
    df = pd.read_csv(cleaned_csv_path)
    booking_col = next((c for c in df.columns if "book" in c.lower()), None)
    delivered_col = next((c for c in df.columns if "deliv" in c.lower()), None)
    loc_col = "section" if "section" in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--groupby", default="section", help="Grouping column name (e.g., section/division)")
    p.add_argument("--metric", default=None, help="Numeric metric column (e.g., delivered, booked, consumption_kwh)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out_dir", default="outputs/plots")
    args = p.parse_args()
    main(args.infile, args.groupby, args.metric, args.top, args.out_dir)
