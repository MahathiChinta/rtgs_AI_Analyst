#!/usr/bin/env python3
"""
analytics.py
- Compute locality/month aggregates from cleaned CSV.
- Identify hotspots (configurable threshold) OR fall back to reliability insights.
- Produce: outputs/analytics-<ts>.json, outputs/evidence-<ts>.json, outputs/report-<ts>.md
- Print a terminal-friendly summary or an ASCII hotspot table.
"""
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def safe_to_int(x):
    try:
        if pd.isna(x) or x=="":
            return np.nan
        return int(float(str(x)))
    except:
        return np.nan

def load_data(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "N/A", None])

def compute_aggregates(df):
    df['tankers_booked'] = df.get('tankers_booked', pd.Series([np.nan]*len(df))).apply(safe_to_int)
    df['tankers_delivered'] = df.get('tankers_delivered', pd.Series([np.nan]*len(df))).apply(safe_to_int)
    df['delivery_gap'] = df.get('delivery_gap', pd.Series([np.nan]*len(df))).apply(safe_to_int)

    if 'month_start' not in df.columns or df['month_start'].isna().all():
        def mk_month(r):
            y = str(r.get('year','')).strip()
            m = str(r.get('month','')).strip()
            if y and m:
                try:
                    mm = int(m)
                except:
                    try:
                        mm = pd.to_datetime(m, format='%b').month
                    except:
                        try:
                            mm = pd.to_datetime(m).month
                        except:
                            return None
                return f"{int(y):04d}-{int(mm):02d}-01"
            return None
        df['month_start'] = df.apply(mk_month, axis=1)

    group = df.groupby(['booking_location','month_start'], dropna=False).agg(
        booked_monthly = ('tankers_booked', lambda s: int(np.nansum(s.values)) if len(s)>0 else 0),
        delivered_monthly = ('tankers_delivered', lambda s: int(np.nansum([v for v in s.values if not pd.isna(v)])) if len(s)>0 else 0)
    ).reset_index()

    try:
        city_median = float(np.nanmedian(group['booked_monthly'].replace(0, np.nan).dropna()))
    except:
        city_median = 0.0

    loc_agg = group.groupby('booking_location', dropna=False).agg(
        total_booked = ('booked_monthly','sum'),
        total_delivered = ('delivered_monthly','sum'),
        months_count = ('month_start','nunique'),
        avg_monthly_booked = ('booked_monthly', lambda s: float(np.nanmean([v for v in s.values if not pd.isna(v)])) if len(s)>0 else 0.0)
    ).reset_index()

    loc_agg['avg_delivery_gap'] = loc_agg.apply(lambda r: (r['total_booked'] - r['total_delivered']) if r['total_booked']>0 else 0, axis=1)
    loc_agg['gap_ratio'] = loc_agg.apply(lambda r: float(r['avg_delivery_gap'])/r['total_booked'] if r['total_booked']>0 else 0.0, axis=1)

    # threshold multiplier can be tweaked
    threshold = max(1.0, city_median * 1.5)
    loc_agg['is_hotspot'] = loc_agg['avg_monthly_booked'].apply(lambda v: True if (v >= threshold and not np.isnan(v)) else False)

    hotspots = loc_agg[loc_agg['is_hotspot']].sort_values('avg_monthly_booked', ascending=False)

    monthly_totals = group.groupby('month_start', dropna=False).agg(
        city_booked = ('booked_monthly','sum'),
        city_delivered = ('delivered_monthly','sum')
    ).reset_index().sort_values('month_start')

    return {
        "city_median_monthly_booked": city_median,
        "threshold_hotspot_avg_monthly_booked": threshold,
        "locality_aggregates": loc_agg.to_dict(orient='records'),
        "hotspots": hotspots.to_dict(orient='records'),
        "monthly_totals": monthly_totals.to_dict(orient='records'),
    }, group, loc_agg

def pick_evidence(df_cleaned, hotspots, max_rows=8):
    evidence = []
    hotspot_localities = [h['booking_location'] for h in hotspots]
    for loc in hotspot_localities:
        subset = df_cleaned[df_cleaned['booking_location']==loc].copy()
        subset['delivery_gap'] = subset.get('delivery_gap', np.nan).apply(safe_to_int)
        subset = subset.sort_values('delivery_gap', ascending=False)
        for _, r in subset.head(3).iterrows():
            evidence.append({
                "row_id": r.get('row_id', None),
                "source_file": r.get('_source_file', None),
                "month_start": r.get('month_start', None),
                "booking_location": r.get('booking_location', None),
                "tankers_booked": int(safe_to_int(r.get('tankers_booked')) or 0),
                "tankers_delivered": int(safe_to_int(r.get('tankers_delivered')) or 0),
                "delivery_gap": int(safe_to_int(r.get('delivery_gap')) or 0)
            })
            if len(evidence) >= max_rows:
                break
        if len(evidence) >= max_rows:
            break
    if len(evidence) < max_rows:
        df_cleaned['delivery_gap'] = df_cleaned.get('delivery_gap', np.nan).apply(safe_to_int)
        topg = df_cleaned.sort_values('delivery_gap', ascending=False).head(max_rows - len(evidence))
        for _, r in topg.iterrows():
            evidence.append({
                "row_id": r.get('row_id', None),
                "source_file": r.get('_source_file', None),
                "month_start": r.get('month_start', None),
                "booking_location": r.get('booking_location', None),
                "tankers_booked": int(safe_to_int(r.get('tankers_booked')) or 0),
                "tankers_delivered": int(safe_to_int(r.get('tankers_delivered')) or 0),
                "delivery_gap": int(safe_to_int(r.get('delivery_gap')) or 0)
            })
    return evidence

def pretty_table(rows, headers):
    col_widths = [max(len(str(h)), max((len(str(r.get(h,''))) for r in rows), default=0)) for h in headers]
    sep = "+".join("-"*(w+2) for w in col_widths)
    lines = []
    header_line = "| " + " | ".join(h.ljust(w) for h,w in zip(headers,col_widths)) + " |"
    lines.append("+" + sep + "+")
    lines.append(header_line)
    lines.append("+" + sep + "+")
    for r in rows:
        line = "| " + " | ".join(str(r.get(h,'')).ljust(w) for h,w in zip(headers,col_widths)) + " |"
        lines.append(line)
    lines.append("+" + sep + "+")
    return "\n".join(lines)

def fallback_insights(df, analytics):
    bk = df.get('tankers_booked', pd.Series(dtype=float)).apply(safe_to_int)
    delv = df.get('tankers_delivered', pd.Series(dtype=float)).apply(safe_to_int)
    gap = df.get('delivery_gap', pd.Series(dtype=float)).apply(safe_to_int)

    total_booked = int(np.nansum([v for v in bk if not pd.isna(v)]) or 0)
    total_gap = int(np.nansum([v for v in gap if not pd.isna(v)]) or 0)
    avg_gap = float(np.nanmean([v for v in gap if not pd.isna(v)]) or 0.0)
    gap_rate = (total_gap / total_booked * 100) if total_booked>0 else 0.0

    monthly = pd.DataFrame(analytics.get('monthly_totals') or [])
    peak_month = None
    if not monthly.empty:
        try:
            monthly['city_booked'] = monthly['city_booked'].astype(float)
            peak = monthly.loc[monthly['city_booked'].idxmax()]
            peak_month = (peak.get('month_start'), float(peak.get('city_booked')))
        except Exception:
            peak_month = None

    return {
        "total_booked": total_booked,
        "total_gap": total_gap,
        "avg_gap": round(avg_gap,2),
        "gap_rate_pct": round(gap_rate,2),
        "peak_month": peak_month
    }

def main(infile, out_dir):
    p = Path(infile)
    if not p.exists():
        print(f"Input {infile} not found.")
        return
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(infile)
    analytics, monthly_group, loc_agg = compute_aggregates(df)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    analytics_file = outdir / f"analytics-{ts}.json"
    evidence_file = outdir / f"evidence-{ts}.json"
    report_file = outdir / f"report-{ts}.md"

    with open(analytics_file, "w", encoding="utf8") as fh:
        json.dump(analytics, fh, indent=2)
    print(f"Wrote analytics to {analytics_file}")

    hotspots = analytics.get('hotspots', [])
    evidence_rows = pick_evidence(df, hotspots, max_rows=8)
    with open(evidence_file, "w", encoding="utf8") as fh:
        json.dump({"evidence": evidence_rows}, fh, indent=2)
    print(f"Wrote evidence to {evidence_file}")

    md_lines = [
        "# RTGS — Hyderabad Water Tanker Analytics",
        "",
        f"**Run timestamp:** {ts}",
        "",
        "## Executive summary",
        ""
    ]

    if hotspots:
        top5 = analytics['hotspots'][:5]
        md_lines.append("Top hotspots (by average monthly bookings):")
        for h in top5:
            md_lines.append(f"- **{h['booking_location']}** — avg monthly booked: {h.get('avg_monthly_booked',0):.1f}, total booked: {h.get('total_booked',0)}, gap ratio: {h.get('gap_ratio',0):.2f}")
    else:
        fb = fallback_insights(df, analytics)
        md_lines.append(f"- **Reliability:** total booked={fb['total_booked']}, total unmet (gap)={fb['total_gap']}, average gap per record={fb['avg_gap']} tankers ({fb['gap_rate_pct']}% of demand).")
        if fb['peak_month']:
            md_lines.append(f"- **Trend:** peak monthly booked in {fb['peak_month'][0]} with {fb['peak_month'][1]} tankers (city total).")
        md_lines.append("- **Policy implication:** System shows good delivery reliability. Maintain monitoring and prioritize proactive measures for identified peak months.")

    md_lines += ["", "## Monthly trend (city totals)", ""]
    for m in analytics.get('monthly_totals', []):
        md_lines.append(f"- {m.get('month_start')}: booked={m.get('city_booked')}, delivered={m.get('city_delivered')}")

    md_lines += ["", "## Evidence (sample rows)", ""]
    for e in evidence_rows:
        md_lines.append(f"- {e['row_id']} | {e['booking_location']} | {e['month_start']} | booked={e['tankers_booked']} | delivered={e['tankers_delivered']} | gap={e['delivery_gap']}")

    with open(report_file, "w", encoding="utf8") as fh:
        fh.write("\n".join(md_lines))
    print(f"Wrote report to {report_file}")

    if analytics.get('hotspots'):
        top5 = analytics['hotspots'][:5]
        rows = []
        for h in top5:
            rows.append({
                "Locality": h.get('booking_location',''),
                "AvgBooked": f"{h.get('avg_monthly_booked',0):.1f}",
                "TotalBooked": int(h.get('total_booked',0)),
                "GapRatio": f"{h.get('gap_ratio',0):.2f}"
            })
        headers = ["Locality","AvgBooked","TotalBooked","GapRatio"]
        print("\nTop hotspots (ASCII table):")
        print(pretty_table(rows, headers))
    else:
        fb = fallback_insights(df, analytics)
        print("\nNo major hotspots detected. Fallback insights:")
        print(f"- Total booked (period): {fb['total_booked']}")
        print(f"- Total unmet (gap): {fb['total_gap']} (avg gap: {fb['avg_gap']} tankers, {fb['gap_rate_pct']}% of demand)")
        if fb['peak_month']:
            print(f"- Peak month: {fb['peak_month'][0]} (city booked = {fb['peak_month'][1]})")
        print("- Policy implication: System shows good delivery reliability; continue monitoring and pre-position resources for peak months.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="data/cleaned/tankers_cleaned.csv")
    parser.add_argument("--out_dir", dest="out_dir", default="outputs")
    args = parser.parse_args()
    main(args.infile, args.out_dir)
