"""Prespecified analysis behavior for the blind training-size extension."""

import pytest

from tools.report_training_size_scale import (
    exact_signflip_p,
    paired_bootstrap_ci,
    terminal_analysis_value,
)


def test_terminal_divergence_policy_preserves_raw_null():
    row = {"status": "diverged", "test_cert_at_T": ""}
    assert terminal_analysis_value(row) == 0.0
    assert row["test_cert_at_T"] == ""
    assert terminal_analysis_value({"status": "completed", "test_cert_at_T": "42.5"}) == 42.5
    with pytest.raises(ValueError, match="raw terminal"):
        terminal_analysis_value({"status": "diverged", "test_cert_at_T": "0"})


def test_exact_signflip_two_sided_and_noop_ties():
    assert exact_signflip_p([1.0, 2.0]) == 0.5
    assert exact_signflip_p([0.0, 0.0, 0.0]) == 1.0
    assert exact_signflip_p([1.0, 0.0]) == 1.0


def test_paired_bootstrap_is_reproducible():
    assert paired_bootstrap_ci([1.0, 2.0, 3.0], draws=100, seed=7) == paired_bootstrap_ci(
        [1.0, 2.0, 3.0], draws=100, seed=7
    )
