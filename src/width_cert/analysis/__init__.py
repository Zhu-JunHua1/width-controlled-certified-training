"""Result aggregation that consumes the public logging schema."""

from .bound_audit import audit_same_box
from .lipschitz_margin import analyze_checkpoint, sampled_lipschitz_margin
from .report import discover_result_files, export_report
from .summary import summarize_results

__all__ = [
    "audit_same_box",
    "analyze_checkpoint",
    "discover_result_files",
    "export_report",
    "sampled_lipschitz_margin",
    "summarize_results",
]
