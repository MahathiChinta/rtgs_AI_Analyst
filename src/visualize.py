
"""
Simple visualization helper for the RTGS pipeline.
Creates:
 - outputs/plots/top_locations_bookings.png
 - outputs/plots/bookings_time_series.png (if time/date columns are present)

Notes:
 - Uses matplotlib and pandas.
 - Does NOT set custom colors per instruction.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def viz_top_locations(df: pd.DataFrame, booking_col: str, loc_col: str, out_path: Path):
    agg = df.groupby(loc_col)[booking_col].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,4))
    agg.plot(kind="bar")
    plt.title("Top 10 locations by bookings")
    plt.ylabel("Bookings")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def viz_time_series(df: pd.DataFrame, booking_col: str, time_col: str, out_path: Path):
    # attempt to parse a date/time column
    try:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        df2 = df.copy()
        df2["__ts__"] = ts.dt.to_period("M").dt.to_timestamp()
        agg = df2.groupby("__ts__")[booking_col].sum().sort_index()
        if len(agg) < 2:
            return False
        plt.figure(figsize=(8,3))
        agg.plot(kind="line", marker="o")
        plt.title("Bookings over time (monthly)")
        plt.ylabel("Bookings")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return True
    except Exception:
        return False

def auto_visualize(cleaned_csv_path: str):
    p = Path(cleaned_csv_path)
    if not p.exists():
        print("visualize.py: cleaned CSV not found:", cleaned_csv_path)
        return
    df = pd.read_csv(p)
    lower_cols = {c.lower(): c for c in df.columns}
    booking_keys = [k for k in lower_cols if "book" in k]
    loc_keys = [k for k in lower_cols if k in ("division","location","section","ward","area")]
    date_keys = [k for k in lower_cols if "date" in k or "month" in k or "year" in k]

    if not booking_keys:
        print("visualize.py: no booking-like column detected; skipping visuals.")
        return

    booking_col = lower_cols[booking_keys[0]]
    # top locations
    if loc_keys:
        loc_col = lower_cols[loc_keys[0]]
        outp = OUT_DIR / f"top_locations_{p.stem}.png"
        viz_top_locations(df, booking_col, loc_col, outp)
        print("Wrote:", outp)

    # time series if possible (prefer explicit date)
    time_col = None
    for candidate in ["date","month","month_iso","year_month","timestamp"]:
        if candidate in lower_cols:
            time_col = lower_cols[candidate]
            break
    if not time_col and date_keys:
        time_col = lower_cols[date_keys[0]]

    if time_col:
        outp2 = OUT_DIR / f"bookings_timeseries_{p.stem}.png"
        ok = viz_time_series(df, booking_col, time_col, outp2)
        if ok:
            print("Wrote:", outp2)
        else:
            print("visualize.py: time series not generated (insufficient time points).")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned/tankers_cleaned_enhanced.csv"
    auto_visualize(path)
