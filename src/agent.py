
"""
agent.py
- Orchestrator that runs ingest -> clean -> analytics (via subprocess).
- Routes outputs: if hotspots found -> produce alerts.json; else produce reliability report.
- Human-in-the-loop (HITL) pause before publishing.
- Writes run trace into traces/run-<ts>.json for observability.
"""
import argparse
import subprocess, json, time, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
TRACE_DIR = ROOT / "traces"
OUTPUTS_DIR = ROOT / "outputs"

def now_ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def run_cmd(cmd):
    try:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        out = proc.stdout.decode("utf8", errors="replace")
        err = proc.stderr.decode("utf8", errors="replace")
        return proc.returncode, out, err
    except Exception as e:
        return 99, "", str(e)

def read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.load(open(p, "r", encoding="utf8"))

def find_latest(prefix):
    files = sorted(OUTPUTS_DIR.glob(prefix + "*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def write_trace(trace):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_ts()
    outp = TRACE_DIR / f"run-{ts}.json"
    json.dump(trace, open(outp, "w", encoding="utf8"), indent=2)
    return outp

def checkpoint_prompt(msg):
    print("\n=== HUMAN CHECKPOINT ===")
    print(msg)
    print("Approve? [y] yes / [n] abort / [s] skip_publish")
    ans = input("> ").strip().lower()
    return ans

def simple_alerts_from_hotspots(analytics_path, evidence_path, out_path):
    analytics = read_json(analytics_path)
    evidence = read_json(evidence_path) or {}
    hotspots = analytics.get("hotspots", []) if analytics else []
    data = {
        "generated_at": now_ts(),
        "hotspots_count": len(hotspots),
        "hotspots": hotspots[:10],
        "evidence_sample": evidence.get("evidence", [])[:8]
    }
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf8")
    return out_path

def reliability_report_from_analytics(analytics_path, evidence_path, out_md_path):
    analytics = read_json(analytics_path) or {}
    evidence = read_json(evidence_path) or {}
    ts = now_ts()
    lines = [f"# Reliability report — RTGS (generated {ts})", ""]
    lines.append(f"- City median monthly booked: {analytics.get('city_median_monthly_booked')}")
    lines.append(f"- Hotspot threshold: {analytics.get('threshold_hotspot_avg_monthly_booked')}")
    lines.append("")
    lines.append("## Monthly totals (sample):")
    for m in analytics.get("monthly_totals", [])[:12]:
        lines.append(f"- {m.get('month_start')}: booked={m.get('city_booked')}, delivered={m.get('city_delivered')}")
    lines.append("")
    lines.append("## Evidence (sample):")
    for e in evidence.get("evidence", [])[:8]:
        lines.append(f"- {e.get('row_id')} | {e.get('booking_location')} | {e.get('month_start')} | booked={e.get('tankers_booked')} | delivered={e.get('tankers_delivered')} | gap={e.get('delivery_gap')}")
    Path(out_md_path).write_text("\n".join(lines), encoding="utf8")
    return out_md_path

def orchestrate(hitl=True, max_retries=1):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    trace = {"run_id": now_ts(), "steps": [], "params": {"hitl": hitl, "max_retries": max_retries}}
    steps = [
        {"name":"ingest", "cmd":"python src/ingest.py --input_dir data/raw --out data/cleaned/tankers_raw_merged.csv"},
        {"name":"clean", "cmd":"python src/clean.py --in data/cleaned/tankers_raw_merged.csv --out data/cleaned/tankers_cleaned.csv"},
        {"name":"analytics", "cmd":"python src/analytics.py --in data/cleaned/tankers_cleaned.csv --out_dir outputs"}
    ]
    for s in steps:
        attempt = 0
        success = False
        rec = {"name": s["name"], "start": now_ts(), "attempts": []}
        while attempt <= max_retries and not success:
            attempt += 1
            rc, out, err = run_cmd(s["cmd"])
            rec["attempts"].append({"attempt": attempt, "rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:], "ts": now_ts()})
            if rc == 0:
                success = True
            else:
                time.sleep(1)
        rec["end"] = now_ts()
        rec["success"] = success
        trace["steps"].append(rec)
        if not success:
            trace["status"] = "failed"
            trace_file = write_trace(trace)
            print(f"Step {s['name']} failed. Trace: {trace_file}")
            return False, trace_file

    analytics_f = find_latest("analytics-")
    evidence_f = find_latest("evidence-")
    trace["artifacts"] = {"analytics": str(analytics_f) if analytics_f else None, "evidence": str(evidence_f) if evidence_f else None}
    analytics = read_json(str(analytics_f)) if analytics_f else {}
    hotspots = analytics.get("hotspots", []) if analytics else []
    route = "alerts" if hotspots else "reliability_report"
    trace["route_decision"] = {"route": route, "hotspots_count": len(hotspots)}

    proceed = "auto"
    if hitl:
        msg = f"Detected {len(hotspots)} hotspots. Approve publishing?" if route=="alerts" else "No hotspots detected. Approve publishing reliability report?"
        ans = checkpoint_prompt(msg)
        if ans == "y":
            proceed = "approved"
        elif ans == "s":
            proceed = "skip_publish"
        else:
            proceed = "abort"
    else:
        proceed = "auto"

    if proceed == "abort":
        trace["status"] = "aborted_by_human"
        trace_file = write_trace(trace)
        print("Aborted by human. Trace:", trace_file)
        return False, trace_file

    ts = now_ts()
    if route == "alerts" and proceed != "skip_publish":
        alerts_path = OUTPUTS_DIR / f"alerts-{ts}.json"
        simple_alerts_from_hotspots(str(analytics_f), str(evidence_f), alerts_path)
        trace["artifacts"]["alerts"] = str(alerts_path)
        action_msg = f"Alerts generated: {alerts_path}"
    else:
        report_path = OUTPUTS_DIR / f"reliability-{ts}.md"
        reliability_report_from_analytics(str(analytics_f), str(evidence_f), report_path)
        trace["artifacts"]["reliability_report"] = str(report_path)
        action_msg = f"Reliability report generated: {report_path}"

    trace["status"] = "success"
    trace_file = write_trace(trace)
    print("Agent run complete. Action:", action_msg)
    print("Trace saved:", trace_file)
    return True, trace_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hitl", action="store_true", default=True)
    parser.add_argument("--no-hitl", dest="hitl", action="store_false")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    ok, tfile = orchestrate(hitl=args.hitl, max_retries=args.max_retries)
    if not ok:
        sys.exit(1)
