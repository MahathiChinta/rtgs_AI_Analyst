"""
Deterministic fallback helper: runs the analytics pipeline and returns a short string
summary when the LLM indicates 'Insufficient evidence'.
"""

from pathlib import Path
from typing import Optional
import json

def deterministic_fallback_summary(cleaned_csv_path: str = "data/cleaned/tankers_cleaned_enhanced.csv") -> str:
    """
    Run deterministic analytics (src/analytics.py main) and return a short textual summary.
    This function imports analytics dynamically so the CLI/QA modules remain lightweight.
    """
    try:
        # dynamic import to avoid circular import issues
        import importlib
        analytics = importlib.import_module("src.analytics")
        res = analytics.main(cleaned_csv_path)
        # read md summary
        report_path = res.get("report")
        if report_path and Path(report_path).exists():
            text = Path(report_path).read_text(encoding="utf-8")
            # return first ~2000 chars so CLI output is readable
            return text[:4000]
        else:
            return "Deterministic analytics ran but no report produced."
    except Exception as e:
        return f"Deterministic fallback failed: {e}"
