"""Apply and verify the auto_LiRPA patch used by the frozen experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


UPSTREAM_COMMIT = "9d100ec070868440b48d34e2f1dd21b97aab9172"
UPSTREAM_TREE_SHA256 = "2bc6390a98d9e88df4c8a2460c9e1cd2e4d7e7e7b058970f665bf7379febff2f"
PATCH_CONTENT_TREE_SHA256 = "4f708bb411c042793f50f5f8d77fbfff07e1baed7a859f87cec15dfe1d8398b1"
PATCHED_TREE_SHA256 = "6a8d960cdd013cbd63cc814d2e21e30ab2f3f23bada889eaf4a91f4be3994c73"


def python_tree_sha256(package_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py"), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_output(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_checkout(source: Path) -> Path:
    package_root = source / "auto_LiRPA"
    if not package_root.is_dir():
        raise SystemExit(f"auto_LiRPA package directory not found under {source}")
    commit = git_output(source, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise SystemExit(f"expected upstream commit {UPSTREAM_COMMIT}, found {commit}")
    return package_root


def restore_historical_whitespace(package_root: Path) -> None:
    """Restore one inert blank line included in the frozen byte-level hash."""

    path = package_root / "operators" / "linear.py"
    content = path.read_bytes()
    marker = (
        b"            ), weight_has_batch=weight_has_batch)\n"
        b"\n"
        b"        # Check if we need to swap x and y\n"
    )
    replacement = (
        b"            ), weight_has_batch=weight_has_batch)\n"
        b"        \n"
        b"        # Check if we need to swap x and y\n"
    )
    if content.count(marker) != 1:
        raise SystemExit("could not locate the frozen historical-whitespace marker")
    path.write_bytes(content.replace(marker, replacement, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="auto_LiRPA Git checkout")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify an already patched checkout without modifying it",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    package_root = require_checkout(source)
    current_hash = python_tree_sha256(package_root)

    if args.check_only:
        if current_hash != PATCHED_TREE_SHA256:
            raise SystemExit(
                f"patched tree hash mismatch: expected {PATCHED_TREE_SHA256}, found {current_hash}"
            )
        print(f"verified patched auto_LiRPA tree: {current_hash}")
        return

    status = git_output(source, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise SystemExit("auto_LiRPA checkout must be clean before applying the publication patch")
    if current_hash != UPSTREAM_TREE_SHA256:
        raise SystemExit(
            f"upstream tree hash mismatch: expected {UPSTREAM_TREE_SHA256}, found {current_hash}"
        )

    repository_root = Path(__file__).resolve().parents[1]
    patch = repository_root / "third_party" / "auto_lirpa" / "historical_runtime.patch"
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "apply",
            "--check",
            "--whitespace=nowarn",
            str(patch),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "apply", "--whitespace=nowarn", str(patch)],
        check=True,
    )

    patch_content_hash = python_tree_sha256(package_root)
    if patch_content_hash != PATCH_CONTENT_TREE_SHA256:
        raise SystemExit(
            "patch-content tree hash mismatch: "
            f"expected {PATCH_CONTENT_TREE_SHA256}, found {patch_content_hash}"
        )
    restore_historical_whitespace(package_root)
    patched_hash = python_tree_sha256(package_root)
    if patched_hash != PATCHED_TREE_SHA256:
        raise SystemExit(
            f"patched tree hash mismatch: expected {PATCHED_TREE_SHA256}, found {patched_hash}"
        )
    print(f"prepared historical auto_LiRPA runtime: {patched_hash}")


if __name__ == "__main__":
    main()
