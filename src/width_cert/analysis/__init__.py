"""Result aggregation that consumes the public logging schema."""

from .bound_audit import audit_same_box
from .lipschitz_margin import analyze_checkpoint, sampled_lipschitz_margin
from .report import discover_result_files, export_report
from .runtime import export_runtime_report
from .soundness_grid import validate_small_network_grid
from .summary import summarize_results
from .statistics import analyze_paired_hypotheses

__all__ = [
    "audit_same_box",
    "analyze_checkpoint",
    "discover_result_files",
    "export_report",
    "export_runtime_report",
    "sampled_lipschitz_margin",
    "summarize_results",
    "analyze_paired_hypotheses",
    "validate_small_network_grid",
]
