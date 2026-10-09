"""Compare two auto_LiRPA Python source trees without recording local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _files(root: Path) -> dict[str, str]:
    records = {}
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        records[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in _files(root).items():
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compare(reference_root: Path, runtime_root: Path) -> dict[str, object]:
    reference = _files(reference_root)
    runtime = _files(runtime_root)
    names = sorted(set(reference) | set(runtime))
    changes = []
    for name in names:
        if reference.get(name) != runtime.get(name):
            changes.append(
                {
                    "path": name,
                    "reference_sha256": reference.get(name),
                    "runtime_sha256": runtime.get(name),
                    "status": (
                        "modified"
                        if name in reference and name in runtime
                        else "runtime_only"
                        if name in runtime
                        else "reference_only"
                    ),
                }
            )
    return {
        "purpose": "dependency provenance comparison; not a soundness proof",
        "reference_python_files": len(reference),
        "runtime_python_files": len(runtime),
        "reference_python_tree_sha256": _tree_hash(reference_root),
        "runtime_python_tree_sha256": _tree_hash(runtime_root),
        "changed_files": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference_root.resolve(), args.runtime_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
