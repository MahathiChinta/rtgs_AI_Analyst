
"""
Simple, explainable forecast per location.

Saves CSV to outputs/forecasts/forecast-next-month-<ts>.csv
and a short markdown brief.

Usage:
 python src/forecast_next_month.py --in data/cleaned/tankers_cleaned_enhanced.csv --out outputs/forecasts
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "forecasts"

def ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    def choose(cands):
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None
    return {
        "date": choose(['date','year_month','month_iso']),
        "bookings": choose(['number_of_bookings','noofbookings','booked','bookings']),
        "location": choose(['location','section','division','locality'])
    }

def prepare_time_index(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['tidx'] = (df[date_col].dt.year*12 + df[date_col].dt.month)  # monotonic integer
    return df

def forecast_for_location(subdf, date_col, target_col):
    # Use simple 3-month moving avg + linear regression as check
    sub = subdf.sort_values(date_col)
    # aggregate monthly
    monthly = sub.groupby(pd.Grouper(key=date_col, freq='M'))[target_col].sum().reset_index()
    if len(monthly) < 2:
        return None, "insufficient_history"
    # moving avg
    ma = monthly[target_col].tail(3).mean()
    # linear regression on numeric idx
    X = np.arange(len(monthly)).reshape(-1,1)
    y = monthly[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    next_idx = np.array([[len(monthly)]])
    pred_lr = float(model.predict(next_idx))
    # choose conservative forecast: avg of MA and LR, not negative
    forecast = max(0, float(np.mean([ma, pred_lr])))
    explanation = f"ma3={ma:.1f}, lr_pred={pred_lr:.1f}"
    return int(round(forecast)), explanation

def main(infile, outdir):
    df = pd.read_csv(infile)
    cols = detect_columns(df)
    if not cols['date'] or not cols['bookings'] or not cols['location']:
        raise SystemExit("Could not detect required columns. Ensure cleaned CSV has date, bookings, location.")
    df = prepare_time_index(df, cols['date'])
    results = []
    for loc, g in df.groupby(cols['location']):
        f, note = forecast_for_location(g, cols['date'], cols['bookings'])
        if f is None:
            continue
        results.append({"location": loc, "forecast_next_period": f, "method_note": note})
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"forecast-next-month-{ts()}.csv"
    md_path = outdir / f"forecast-brief-{ts()}.md"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    # write short brief
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Forecast brief\n\n")
        fh.write(f"Forecast run: {ts()}\n\n")
        fh.write("Sample forecasts (location, forecast_next_period, method_note)\n\n")
        for r in results[:20]:
            fh.write(f"- {r['location']}: {r['forecast_next_period']} ({r['method_note']})\n")
    print("Wrote forecast CSV:", csv_path)
    print("Wrote brief:", md_path)
    return csv_path, md_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outdir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    main(args.infile, args.outdir)
