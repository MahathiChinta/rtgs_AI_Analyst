
"""
Deterministic fallback: simple, evidence-first summaries computed
directly from the cleaned CSV. Used when LLM claims 'Insufficient evidence'
so the CLI still prints something useful.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict

TS = lambda: datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def deterministic_fallback_summary(cleaned_csv_path: str) -> str:
    p = Path(cleaned_csv_path)
    if not p.exists():
        return "Deterministic fallback: cleaned CSV not found."

    df = pd.read_csv(p)
    # try to robustly find numeric booking & delivery columns
    lower_cols = {c.lower(): c for c in df.columns}
    # heuristics
    booking_keys = [k for k in lower_cols if "book" in k]
    delivered_keys = [k for k in lower_cols if "deliver" in k or "suppl" in k]
    location_keys = [k for k in lower_cols if k in ("division","location","section","ward","area")]

    if not booking_keys:
        return "Deterministic fallback: no booking-like column detected in cleaned CSV."

    bk = df[lower_cols[booking_keys[0]]].astype(float)
    out_lines = []
    out_lines.append(f"Deterministic summary (from {p.name}):")
    out_lines.append(f"- Total rows: {len(df):,}")
    out_lines.append(f"- Booking column used: {lower_cols[booking_keys[0]]}")
    out_lines.append(f"- Sum of bookings: {int(bk.sum()):,}")
    out_lines.append(f"- Median bookings per row: {int(bk.median())}")
    # top locations by bookings if loc available
    if location_keys:
        loc_col = lower_cols[location_keys[0]]
        grp = df.groupby(loc_col)[lower_cols[booking_keys[0]]].sum().sort_values(ascending=False).head(6)
        out_lines.append("- Top locations by bookings (top 6):")
        for loc, val in grp.items():
            out_lines.append(f"  * {loc} — {int(val):,} bookings")
    # compute delivery gap if delivery column exists
    if delivered_keys:
        dv = df[lower_cols[delivered_keys[0]]].astype(float)
        gap = (bk - dv)
        out_lines.append(f"- Overall delivered sum: {int(dv.sum()):,}")
        out_lines.append(f"- Total unmet (booked - delivered): {int(gap.sum()):,}")
        # show any hotspots
        hotspots = gap[gap > 0].sort_values(ascending=False).head(5)
        if len(hotspots):
            out_lines.append("- Hotspots (rows with largest positive booked-delivered):")
            for idx, g in hotspots.iteritems():
                row = df.iloc[int(idx)]
                loc = row.get(loc_col, "N/A") if location_keys else "N/A"
                out_lines.append(f"  * row {idx} ({loc}) — gap {int(g)}")
        else:
            out_lines.append("- No positive booked-delivered gaps detected in the data.")
    else:
        out_lines.append("- No delivery-like column detected to compute booked-delivered gaps.")

    return "\n".join(out_lines)


if __name__ == "__main__":
    # quick smoke test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned/tankers_cleaned_enhanced.csv"
    print(deterministic_fallback_summary(path))
