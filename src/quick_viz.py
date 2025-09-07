"""
Quick visualizations for RTGS pipeline.

Produces PNGs in outputs/figs:
 - top_locations_by_bookings.png
 - bookings_vs_delivered_trend.png
 - monthly_trend_top_locations.png

Usage:
 python src/quick_viz.py --in data/cleaned/tankers_cleaned_enhanced.csv --out outputs/figs
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "figs"

def ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def find_columns(df: pd.DataFrame):
    # robust column detection: common variants
    col_map = {}
    cols = [c.lower() for c in df.columns]
    def choose(cands):
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None
    col_map['bookings'] = choose(['number_of_bookings','noofbookings','booked','bookings','no_of_bookings'])
    col_map['delivered'] = choose(['delivered_count','delivered','delivered_count','delivered_qty'])
    col_map['date'] = choose(['date','year_month','month_iso','month','year_month_date'])
    col_map['location'] = choose(['location','section','division','locality','area','ward'])
    return col_map

def ensure_date(df, date_col):
    if date_col is None:
        return df
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    return df

def top_locations_by_bookings(df, col_map, outdir: Path):
    bcol = col_map['bookings']
    loc = col_map['location']
    if not bcol or not loc:
        return None
    agg = df.groupby(loc)[bcol].sum().sort_values(ascending=False).head(10)
    out = outdir / f"top_locations_by_bookings-{ts()}.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    agg.plot.barh()
    plt.gca().invert_yaxis()
    plt.xlabel("Total bookings")
    plt.title("Top 10 locations by total bookings")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def bookings_vs_delivered_trend(df, col_map, outdir: Path):
    bcol = col_map['bookings']
    dcol = col_map['delivered']
    datecol = col_map['date']
    if not bcol or not dcol or datecol is None:
        return None
    df = ensure_date(df, datecol)
    df2 = df.groupby(pd.Grouper(key=datecol, freq='M'))[[bcol,dcol]].sum().reset_index()
    out = outdir / f"bookings_vs_delivered_trend-{ts()}.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(df2[datecol], df2[bcol], label="bookings")
    plt.plot(df2[datecol], df2[dcol], label="delivered")
    plt.legend()
    plt.xlabel("Month")
    plt.title("Bookings vs Delivered (monthly)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def monthly_trend_top_locations(df, col_map, outdir: Path, top_n=5):
    bcol = col_map['bookings']; datecol = col_map['date']; loc = col_map['location']
    if not (bcol and datecol and loc):
        return None
    df = ensure_date(df, datecol)
    agg = df.groupby([loc, pd.Grouper(key=datecol, freq='M')])[bcol].sum().reset_index()
    toplocs = agg.groupby(loc)[bcol].sum().nlargest(top_n).index.tolist()
    out = outdir / f"monthly_trend_top_locations-{ts()}.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,5))
    for l in toplocs:
        sub = agg[agg[loc]==l]
        plt.plot(sub[datecol], sub[bcol], label=str(l))
    plt.legend()
    plt.xlabel("Month")
    plt.title(f"Monthly bookings for top {top_n} locations")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def main(infile, outdir):
    df = pd.read_csv(infile)
    col_map = find_columns(df)
    results = {}
    outdir = Path(outdir)
    results['top_locations_png'] = top_locations_by_bookings(df, col_map, outdir)
    results['bookings_vs_delivered_png'] = bookings_vs_delivered_trend(df, col_map, outdir)
    results['monthly_trend_png'] = monthly_trend_top_locations(df, col_map, outdir)
    print("Wrote visuals:", results)
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outdir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args.infile, args.outdir)
