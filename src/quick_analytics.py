# src/quick_analytics.py
import pandas as pd
from pathlib import Path
import sys

def main(csv_path="data/cleaned/tankers_cleaned_enhanced.csv"):
    p = Path(csv_path)
    if not p.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(p)
    # make sure 'year' and 'month' and 'noofbookings' exist
    for c in ("year","month","noofbookings"):
        if c not in df.columns:
            print(f"Missing expected column: {c}")
            print("Columns available:", df.columns.tolist())
            sys.exit(1)
    # convert to numeric
    df["noofbookings"] = pd.to_numeric(df["noofbookings"], errors="coerce").fillna(0).astype(int)
    # group and sort
    agg = df.groupby(["year","month"], as_index=False)["noofbookings"].sum()
    agg = agg.sort_values(["year","month"])
    # Print simple ASCII table and highlight top
    print("\nMonthly booking totals (from cleaned CSV):\n")
    print(agg.to_string(index=False))
    top = agg.sort_values("noofbookings", ascending=False).head(5)
    print("\nTop months by bookings (top 5):")
    print(top.to_string(index=False))
    # Simple implication text:
    if len(agg) == 0:
        print("\nNo data to compute.")
    else:
        max_row = agg.loc[agg["noofbookings"].idxmax()]
        print(f"\nImplication: The month with highest observed bookings is {int(max_row['month'])}/{int(max_row['year'])} with {int(max_row['noofbookings'])} bookings. Policymakers should consider redistributing tanker capacity into that period or increasing standby resources for that month (if representative).")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/cleaned/tankers_cleaned_enhanced.csv")
    args = p.parse_args()
    main(args.csv)
