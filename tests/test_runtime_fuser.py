"""Runtime policy is explicit and fails closed on affected Windows CUDA hosts."""

from types import SimpleNamespace

import pytest
import torch

from width_cert.experiments import environment


def test_cpu_keeps_default_runtime_policy():
    assert environment.configure_bound_runtime(torch.device("cpu")) == "default"


def test_windows_cuda_disables_gpu_jit_fusion(monkeypatch):
    state = {"enabled": True}
    fake_c = SimpleNamespace(
        _jit_override_can_fuse_on_gpu=lambda value: state.update(enabled=value),
        _jit_can_fuse_on_gpu=lambda: state["enabled"],
    )
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment.torch, "_C", fake_c)
    assert environment.configure_bound_runtime(torch.device("cuda")) == (
        "windows_cuda_jit_gpu_fusion_disabled"
    )
    assert state["enabled"] is False


def test_windows_cuda_rejects_missing_gpu_fuser_controls(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment.torch, "_C", SimpleNamespace())
    with pytest.raises(RuntimeError, match="GPU-fuser controls"):
        environment.configure_bound_runtime(torch.device("cuda"))
