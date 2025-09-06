
import pandas as pd
from pathlib import Path
from datetime import datetime

infile = Path("data/cleaned/tankers_raw_merged.csv")
out = Path("data/cleaned/tankers_raw_merged_fixed.csv")

df = pd.read_csv(infile, dtype=str, keep_default_na=False, na_values=["", "NA", "N/A"])
# Try to build a date from available columns
if "date" not in df.columns:
    if "year" in df.columns and "month" in df.columns:
        def make_date(r):
            y = r.get("year")
            m = r.get("month")
            try:
                y = int(y); m = int(m)
                return f"{y:04d}-{m:02d}-01"
            except:
                return ""
        df["date"] = df.apply(make_date, axis=1)
    else:
        # fallback: use current run date
        df["date"] = datetime.utcnow().strftime("%Y-%m-%d")

# add id if missing
if "id" not in df.columns:
    df.insert(0, "id", [f"r{idx+1}" for idx in range(len(df))])

df.to_csv(out, index=False)
print("Wrote", out, "rows=", len(df))
