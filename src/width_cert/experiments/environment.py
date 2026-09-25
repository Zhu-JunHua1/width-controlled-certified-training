"""Reproducibility metadata collection."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import torch


AUTO_LIRPA_REFERENCE_COMMIT = "9d100ec070868440b48d34e2f1dd21b97aab9172"
AUTO_LIRPA_REFERENCE_TREE_SHA256 = (
    "2bc6390a98d9e88df4c8a2460c9e1cd2e4d7e7e7b058970f665bf7379febff2f"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_tree_sha256(package_file: str | Path) -> str:
    """Hash Python sources by relative path and bytes for dependency provenance."""

    root = Path(package_file).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_metadata() -> dict[str, object]:
    try:
        import auto_LiRPA
        lirpa_version = getattr(auto_LiRPA, "__version__", None)
        lirpa_path = getattr(auto_LiRPA, "__file__", None)
        lirpa_hash = package_tree_sha256(lirpa_path) if lirpa_path else None
    except ImportError:
        lirpa_version = lirpa_path = lirpa_hash = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "auto_lirpa_version": lirpa_version,
        "auto_lirpa_module": "auto_LiRPA" if lirpa_path else None,
        "auto_lirpa_source_commit": (
            AUTO_LIRPA_REFERENCE_COMMIT
            if lirpa_hash == AUTO_LIRPA_REFERENCE_TREE_SHA256
            else None
        ),
        "auto_lirpa_python_tree_sha256": lirpa_hash,
    }
