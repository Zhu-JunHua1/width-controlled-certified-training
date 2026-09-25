"""Terminal progress reporting and structured file logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .checkpoints import save_checkpoint
from .records import SCHEMA_VERSION, StateRecord, TrainingRun, UpdateRecord


_MAJOR_RULE = "=" * 118
_MINOR_RULE = "-" * 118


def _percentage(value: float | None) -> str:
    return "       n/a" if value is None else f"{value:9.2f}%"


def _scientific(value: float | None, width: int = 12) -> str:
    return "n/a".rjust(width) if value is None else f"{value:{width}.5e}"


class ConsoleReporter:
    """Readable experiment-level progress with explicit stage boundaries."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def experiment_start(
        self,
        *,
        name: str,
        device: str,
        candidate_start: int,
        candidate_stop: int,
        accepted_seeds: int,
        radii: tuple[float, ...],
        methods: tuple[str, ...],
        total_cells: int,
    ) -> None:
        if not self.enabled:
            return
        print(_MAJOR_RULE)
        print(f"[EXPERIMENT] {name}")
        print(
            f"device={device} | candidate_seeds=[{candidate_start}, {candidate_stop}) "
            f"| accepted_seeds={accepted_seeds} | radii={list(radii)}"
        )
        print(f"methods={list(methods)} | total_training_cells={total_cells}")
        print(_MAJOR_RULE)

    def selection_start(
        self,
        *,
        candidate_start: int,
        candidate_stop: int,
        requested: int,
        pretraining_enabled: bool,
        target_accuracy: float,
    ) -> None:
        if not self.enabled:
            return
        target = f"{100.0 * target_accuracy:.2f}%" if pretraining_enabled else "from scratch"
        print(
            f"[SEED SELECTION START] candidates=[{candidate_start}, {candidate_stop}) "
            f"| requested={requested} | target={target}"
        )
        print(_MINOR_RULE)

    def selection_candidate(
        self,
        *,
        candidate_index: int,
        candidate_total: int,
        seed: int,
        accepted: bool,
        initial_accuracy: float,
        prepared_accuracy: float | None = None,
        pretraining_epochs: int | None = None,
        reason: str = "",
    ) -> None:
        if not self.enabled:
            return
        progress = f"{candidate_index:02d}/{candidate_total:02d}"
        decision = "ACCEPT" if accepted else "REJECT"
        prepared = _percentage(
            None if prepared_accuracy is None else 100.0 * prepared_accuracy
        )
        epochs = "n/a" if pretraining_epochs is None else str(pretraining_epochs)
        detail = reason.replace("_", " ") if reason else ""
        line = (
            f"[{progress}] seed={seed:4d} | {decision:6s} | "
            f"initial_acc={_percentage(100.0 * initial_accuracy)} | "
            f"prepared_acc={prepared} | epochs={epochs:>5s}"
        )
        if detail:
            line += f" | reason={detail}"
        print(line)

    def selection_complete(self, *, selected: list[int], rejected: int) -> None:
        if not self.enabled:
            return
        print(_MINOR_RULE)
        print(
            f"[SEED SELECTION COMPLETE] selected={selected} | "
            f"accepted={len(selected)} | rejected={rejected}"
        )

    def seed_start(
        self,
        *,
        index: int,
        total: int,
        seed: int,
        prepared_accuracy: float,
        pretraining_epochs: int,
    ) -> None:
        if not self.enabled:
            return
        print()
        print(_MAJOR_RULE)
        print(
            f"[SEED {index:02d}/{total:02d} START] seed={seed} | "
            f"prepared_acc={100.0 * prepared_accuracy:.2f}% | "
            f"pretraining_epochs={pretraining_epochs}"
        )
        print(_MAJOR_RULE)

    def seed_complete(self, *, index: int, total: int, seed: int, completed_cells: int) -> None:
        if not self.enabled:
            return
        print(
            f"[SEED {index:02d}/{total:02d} COMPLETE] seed={seed} "
            f"| completed_training_cells={completed_cells}"
        )

    def seed_summary(self, *, seed: int, rows: list[dict[str, Any]]) -> None:
        """Print all requested method results for one shared seed."""

        if not self.enabled or not rows:
            return
        print()
        print(f"[SEED SUMMARY] seed={seed}")
        print(
            " train_eps | method     | peak_cert | peak_epoch | cert_at_T | clean_at_T | status      | IWAU_status"
        )
        print(
            "-----------+------------+-----------+------------+-----------+------------+-------------+---------------"
        )
        for row in rows:
            iwau_status = str(row.get("iwau_status", "active" if row["iwau_effective"] else "n/a"))
            print(
                f" {float(row['radius']):9g} | {str(row['method']):10s} | "
                f"{_percentage(float(row['peak_certified_accuracy']))} | "
                f"{int(row['peak_epoch']):10d} | {_percentage(row['cert_at_T'])} | "
                f"{_percentage(row['final_clean_accuracy'])} | "
                f"{str(row['status']):11s} | {iwau_status:13s}"
            )

    def experiment_summary(self, *, rows: list[dict[str, Any]]) -> None:
        """Print mean and population standard deviation across accepted seeds."""

        if not self.enabled or not rows:
            return
        groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (float(row["radius"]), str(row["method"]))
            groups.setdefault(key, []).append(row)
        print()
        print("[EXPERIMENT SUMMARY] mean +/- population std across accepted seeds")
        print(
            " train_eps | method     | seeds |       peak_cert |        cert_at_T | endpoint_n | completed | IWAU_active"
        )
        print(
            "-----------+------------+-------+-----------------+------------------+------------+-----------+------------"
        )
        for (radius, method), group in groups.items():
            peaks = [float(row["peak_certified_accuracy"]) for row in group]
            endpoints = [float(row["cert_at_T"]) for row in group if row["cert_at_T"] is not None]
            peak_mean = fmean(peaks)
            peak_std = pstdev(peaks) if len(peaks) > 1 else 0.0
            if endpoints:
                endpoint_mean = fmean(endpoints)
                endpoint_std = pstdev(endpoints) if len(endpoints) > 1 else 0.0
                endpoint_text = f"{endpoint_mean:6.2f}+/-{endpoint_std:5.2f}%"
            else:
                endpoint_text = "n/a"
            completed = sum(row["status"] == "completed" for row in group)
            active = sum(bool(row["iwau_effective"]) for row in group)
            print(
                f" {radius:9g} | {method:10s} | {len(group):5d} | "
                f"{peak_mean:6.2f}+/-{peak_std:5.2f}% | {endpoint_text:>16s} | "
                f"{len(endpoints):10d} | {completed:9d} | {active:11d}"
            )

    def experiment_complete(self, *, name: str, output_dir: Path, completed_cells: int) -> None:
        if not self.enabled:
            return
        print()
        print(_MAJOR_RULE)
        print(
            f"[EXPERIMENT COMPLETE] {name} | completed_training_cells={completed_cells}"
        )
        print(f"results={output_dir}")
        print(_MAJOR_RULE)


class RunLogger:
    """Persist configuration, states, updates, summary, and terminal status."""

    def __init__(self, output_dir: str | Path, *, echo: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._epoch_header_printed = False

    def begin_run(
        self,
        *,
        index: int,
        total: int,
        method: str,
        seed: int,
        training_radius: float,
        evaluation_radius: float,
        horizon: int,
    ) -> None:
        if not self.echo:
            return
        print()
        print(_MINOR_RULE)
        print(
            f"[RUN {index:03d}/{total:03d} START] method={method:<9s} | seed={seed:4d} | "
            f"train_eps={training_radius:g} | eval_eps={evaluation_radius:g} | horizon={horizon}"
        )
        print(f"output={self.output_dir}")
        print(_MINOR_RULE)

    def echo_calibration(
        self,
        *,
        source_method: str,
        status: str,
        peak_epoch: int,
        reference_epoch: int | None,
        coefficient: float,
    ) -> None:
        if not self.echo:
            return
        reference = "n/a" if reference_epoch is None else str(reference_epoch)
        print(
            f"[IWAU CALIBRATION] source={source_method} | status={status} | "
            f"peak_epoch={peak_epoch} | reference_epoch={reference} | "
            f"coefficient={coefficient:.6e}"
        )

    def echo_internal_prerun_start(self, *, method: str, horizon: int) -> None:
        if not self.echo:
            return
        print(
            f"[CALIBRATION PRE-RUN] method={method}-fixed | horizon={horizon} | "
            "role=internal dependency | reported_result=no"
        )

    def echo_internal_prerun_complete(
        self, *, run: TrainingRun, elapsed_seconds: float
    ) -> None:
        if not self.echo:
            return
        print(
            f"[CALIBRATION PRE-RUN COMPLETE] peak_cert={run.peak_certified_accuracy:.2f}% "
            f"(epoch={run.peak_epoch}) | cert_at_T={_percentage(run.cert_at_horizon).strip()} "
            f"| elapsed={elapsed_seconds:.2f}s"
        )

    def _echo_epoch_header(self) -> None:
        if self._epoch_header_printed:
            return
        print(
            "  epoch |   cert_acc |  clean_acc | weight_width |   grad_width |  mean_step | update"
        )
        print(
            "--------+------------+------------+--------------+--------------+------------+--------"
        )
        self._epoch_header_printed = True

    def write_config(self, config: dict[str, Any]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **config}
        self._write_json("config.json", payload)

    def epoch_callback(self, state: StateRecord, update: UpdateRecord | None) -> None:
        if not self.echo:
            return
        self._echo_epoch_header()
        gradient_width = None if update is None else update.gradient_width
        mean_step = None if update is None else update.effective_step_mean
        update_name = "initial" if update is None else ("IWAU" if update.iwau_active else "fixed")
        print(
            f"{state.epoch:7d} | {_percentage(state.certified_accuracy)} | "
            f"{_percentage(state.clean_accuracy)} | {_scientific(state.weight_width)} | "
            f"{_scientific(gradient_width)} | {_scientific(mean_step, 10)} | {update_name:<7s}"
        )

    def echo_run(self, run: TrainingRun) -> None:
        """Print an already-completed run using the same per-epoch format."""

        if not self.echo:
            return
        updates = {record.next_epoch: record for record in run.updates}
        for state in run.states:
            self.epoch_callback(state, updates.get(state.epoch))

    def finish_run(self, run: TrainingRun, *, elapsed_seconds: float) -> None:
        if not self.echo:
            return
        final_state = run.states[-1]
        print(_MINOR_RULE)
        print(
            f"[RUN COMPLETE] status={run.status.value} | "
            f"peak_cert={run.peak_certified_accuracy:.2f}% (epoch={run.peak_epoch}) | "
            f"cert_at_T={_percentage(run.cert_at_horizon).strip()} | "
            f"final_clean_acc={_percentage(final_state.clean_accuracy).strip()} | "
            f"elapsed={elapsed_seconds:.2f}s"
        )
        if run.status_reason:
            print(f"status_reason={run.status_reason.replace('_', ' ')}")

    def fail_run(self, error: BaseException) -> None:
        if not self.echo:
            return
        print(_MINOR_RULE)
        print(f"[RUN FAILED] {type(error).__name__}: {error}")

    def write_run(self, run: TrainingRun, *, extra_summary: dict[str, Any] | None = None) -> None:
        self._write_csv("states.csv", run.state_rows())
        self._write_csv("updates.csv", run.update_rows())
        save_checkpoint(
            self.output_dir / "last_state.pt",
            epoch=run.states[-1].epoch,
            state=run.final_state,
        )
        if run.peak_epoch in run.snapshots:
            save_checkpoint(
                self.output_dir / "peak_state.pt",
                epoch=run.peak_epoch,
                state=run.snapshots[run.peak_epoch],
            )
        summary = run.summary()
        if extra_summary:
            summary.update(extra_summary)
        self._write_json("summary.json", summary)
        self.write_terminal_status(
            status=run.status.value,
            reason=run.status_reason,
            cert_at_T=run.cert_at_horizon,
            last_observed_certified_accuracy=run.states[-1].certified_accuracy,
        )

    def write_terminal_status(
        self,
        *,
        status: str,
        reason: str,
        cert_at_T: float | None = None,
        last_observed_certified_accuracy: float | None = None,
        finalized: bool = True,
    ) -> None:
        self._write_json(
            "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "reason": reason,
                "finalized": finalized,
                "cert_at_T": cert_at_T,
                "last_observed_certified_accuracy": last_observed_certified_accuracy,
            },
        )

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self.output_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
