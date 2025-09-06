
"""
Deterministic analytics & visualizations for RTGS tanker dataset.
Input: cleaned CSV (analysis-ready).
Outputs:
 - CSV summaries in outputs/
 - Plots in outputs/plots/
 - A small Markdown summary in outputs/analytics_summary.md
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

BASE = Path(".")
OUT = BASE / "outputs"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def load_cleaned(path="data/cleaned/tankers_cleaned_enhanced.csv"):
    df = pd.read_csv(path)
    return df

def ensure_numeric(df):
    for c in ["number_of_bookings", "delivered_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def compute_summary(df):
    # delivery gap and ratio
    df["gap"] = df.get("number_of_bookings", 0) - df.get("delivered_count", 0)
    df["delivery_ratio"] = df.apply(lambda r: r["delivered_count"] / r["number_of_bookings"] 
                                    if r["number_of_bookings"]>0 else 1.0, axis=1)
    return df

def top_localities(df, n=10):
    agg = df.groupby("section").agg(
        total_bookings=("number_of_bookings","sum"),
        total_delivered=("delivered_count","sum"),
        avg_ratio=("delivery_ratio","mean")
    ).sort_values("total_bookings", ascending=False).reset_index()
    return agg.head(n)

def monthly_trend(df):
    if "year" in df.columns and "month" in df.columns:
        grp = df.groupby(["year","month"]).agg(total_bookings=("number_of_bookings","sum")).reset_index()
        # build a "yyyymm" str for plotting
        grp["yyyymm"] = grp["year"].astype(str) + "-" + grp["month"].astype(str).str.zfill(2)
        return grp
    return None

def save_csv(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    return p

def plot_top_localities(agg, fname=None):
    fname = fname or (PLOTS / f"top_localities_{TS()}.png")
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(agg["section"].astype(str), agg["total_bookings"][::-1])
    ax.set_xlabel("Total bookings")
    ax.set_title("Top localities by bookings")
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fname

def plot_delivery_ratio_hist(df, fname=None):
    fname = fname or (PLOTS / f"delivery_ratio_hist_{TS()}.png")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["delivery_ratio"].dropna(), bins=20)
    ax.set_xlabel("Delivery ratio (delivered / bookings)")
    ax.set_title("Distribution of delivery ratios")
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fname

def produce_report(top_df, monthly_df, gaps_df, charts, out_md="outputs/analytics_summary.md"):
    lines = []
    lines.append("# Data-driven analytics summary")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
    lines.append("## Top localities by total bookings\n")
    lines.append(top_df.to_markdown(index=False))
    lines.append("\n## Notable delivery gaps (positive gap means unmet demand)\n")
    if gaps_df is not None and not gaps_df.empty:
        lines.append(gaps_df.to_markdown(index=False))
    else:
        lines.append("No large gaps detected in supplied dataset.")
    if monthly_df is not None:
        lines.append("\n## Monthly bookings trend (sample)\n")
        lines.append(monthly_df.head(12).to_markdown(index=False))
    lines.append("\n## Plots\n")
    for c in charts:
        lines.append(f"- {c}\n")
    Path(out_md).write_text("\n\n".join(lines), encoding="utf-8")
    return out_md

def main(infile="data/cleaned/tankers_cleaned_enhanced.csv"):
    df = load_cleaned(infile)
    df = ensure_numeric(df)
    df = compute_summary(df)
    # hotspots: where gap > 0 (unmet) or low delivery ratio
    gaps = df[df["gap"] > 0].groupby("section").agg(
        total_gap=("gap","sum"),
        bookings=("number_of_bookings","sum"),
        delivered=("delivered_count","sum")
    ).reset_index().sort_values("total_gap", ascending=False)
    top = top_localities(df, n=12)
    monthly = monthly_trend(df)
    c1 = plot_top_localities(top)
    c2 = plot_delivery_ratio_hist(df)
    save_csv(top, f"top_localities_{TS()}.csv")
    if not gaps.empty:
        save_csv(gaps, f"gaps_{TS()}.csv")
    report = produce_report(top, monthly, gaps.head(10) if not gaps.empty else None, [c1,c2])
    print("Analytics done. Report:", report)
    return {"report": report, "plots": [str(c1), str(c2)]}

if __name__ == "__main__":
    main()
