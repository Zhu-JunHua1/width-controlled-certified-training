"""Unit tests for scalar and coordinate-wise fairness baselines."""

from __future__ import annotations

import math

import torch

from width_cert.core import (
    CertifiedTrainer,
    CosineDecayUpdate,
    FixedUpdate,
    GlobalWidthAdaptiveUpdate,
    InverseTimeUpdate,
    IntervalWidthAdaptiveUpdate,
    StepDecayUpdate,
    budget_record,
    continue_from_fixed_prefix,
    cumulative_matched_cosine_rule,
    final_continuation_step,
    matched_final_step_rules,
)


LOWER = torch.tensor([-2.0, -1.0])
UPPER = torch.tensor([1.0, 3.0])
WIDTH = UPPER - LOWER


def test_inverse_time_uses_one_based_post_prefix_index():
    rule = InverseTimeUpdate(base_step=0.1, start_epoch=4, alpha=1.0)
    first = rule.decide(LOWER, UPPER, WIDTH, epoch=4)
    second = rule.decide(LOWER, UPPER, WIDTH, epoch=5)
    assert math.isclose(first.effective_step_mean, 0.05)
    assert math.isclose(second.effective_step_mean, 0.1 / 3.0)
    assert first.rule_name == "inverse_time"


def test_cosine_reaches_declared_minimum_at_final_continuation_update():
    rule = CosineDecayUpdate(
        base_step=0.1,
        minimum_step=0.02,
        start_epoch=3,
        total_steps=4,
    )
    final = rule.decide(LOWER, UPPER, WIDTH, epoch=6)
    assert math.isclose(final.effective_step_mean, 0.02)


def test_step_decay_switches_at_declared_milestone():
    rule = StepDecayUpdate(base_step=0.1, start_epoch=2, milestone=3, factor=0.25)
    assert math.isclose(rule.decide(LOWER, UPPER, WIDTH, 3).effective_step_mean, 0.1)
    assert math.isclose(rule.decide(LOWER, UPPER, WIDTH, 4).effective_step_mean, 0.025)


def test_global_width_uses_one_scalar_while_iwau_is_coordinate_wise():
    coefficient = 2.0
    global_rule = GlobalWidthAdaptiveUpdate(0.1, coefficient)
    coordinate_rule = IntervalWidthAdaptiveUpdate(0.1, coefficient)
    global_decision = global_rule.decide(LOWER, UPPER, WIDTH, 0)
    coordinate_decision = coordinate_rule.decide(LOWER, UPPER, WIDTH, 0)

    expected_global = 0.1 / (1.0 + coefficient * float(WIDTH.mean()))
    assert math.isclose(global_decision.scalar_step, expected_global)
    assert torch.equal(global_decision.lower_gradient, LOWER)
    assert coordinate_decision.scalar_step == 1.0
    assert not torch.equal(coordinate_decision.lower_gradient, LOWER)
    assert coordinate_decision.rule_name == "IWAU"


def test_matched_scalar_schedules_share_the_requested_final_mean_step():
    rules = matched_final_step_rules(
        base_step=0.1,
        target_final_step=0.02,
        start_epoch=3,
        total_steps=5,
    )
    for rule in rules.values():
        decision = rule.decide(LOWER, UPPER, WIDTH, epoch=7)
        assert math.isclose(decision.effective_step_mean, 0.02, rel_tol=1e-6)


def test_cumulative_cosine_matches_declared_mean_step():
    target = 0.06
    rule = cumulative_matched_cosine_rule(
        base_step=0.1,
        target_mean_step=target,
        start_epoch=3,
        total_steps=8,
    )
    realized = [
        rule.decide(LOWER, UPPER, WIDTH, epoch).effective_step_mean
        for epoch in range(3, 11)
    ]
    assert math.isclose(sum(realized) / len(realized), target, rel_tol=1e-12)


def test_cumulative_cosine_rejects_infeasible_target():
    try:
        cumulative_matched_cosine_rule(
            base_step=0.1,
            target_mean_step=0.01,
            start_epoch=0,
            total_steps=20,
        )
    except ValueError as error:
        assert "infeasible" in str(error)
    else:
        raise AssertionError("infeasible cumulative target was accepted")


def test_common_prefix_continuation_and_budget_are_exact():
    def gradient_bounds(_state):
        return torch.tensor([-1.0]), torch.tensor([1.0])

    def apply_update(state, lower, upper, step):
        state["lower"] -= step * upper
        state["upper"] -= step * lower

    trainer = CertifiedTrainer(
        gradient_bounds=gradient_bounds,
        certified_accuracy=lambda state: 100.0 - float(state["upper"] - state["lower"]),
        apply_update=apply_update,
        horizon=5,
    )
    initial = {"lower": torch.zeros(1), "upper": torch.zeros(1)}
    fixed = trainer.run(initial, FixedUpdate(0.1), capture_snapshots=True)
    continued = continue_from_fixed_prefix(
        trainer,
        fixed,
        start_epoch=2,
        update_rule=InverseTimeUpdate(0.1, start_epoch=2, alpha=1.0),
    )
    assert continued.states[:3] == fixed.states[:3]
    assert [update.update_rule for update in continued.updates] == [
        "fixed",
        "fixed",
        "inverse_time",
        "inverse_time",
        "inverse_time",
    ]
    assert math.isclose(final_continuation_step(continued, start_epoch=2), 0.025)
    budget = budget_record(fixed, continued, start_epoch=2)
    assert budget.fixed_pre_run_updates == 5
    assert budget.continuation_updates == 3
    assert budget.total_bound_evaluations == 8
