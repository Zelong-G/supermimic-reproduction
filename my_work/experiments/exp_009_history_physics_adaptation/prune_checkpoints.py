"""Explicit retention plan for completed six-hour run; dry-run unless --apply."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from common import OUT, ROOT, file_hash, output_path, save_json

KEEP = {5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 80862}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    directory = output_path(OUT / "joint_6h_from5000")
    files = sorted(directory.glob("ckpt_*.pt"))
    keep = []
    remove = []
    for path in files:
        token = path.stem.removeprefix("ckpt_")
        if not token.isdigit() or path.is_symlink():
            raise ValueError(f"Unexpected checkpoint: {path}")
        (keep if int(token) in KEEP else remove).append(path)
    assert {int(p.stem.split("_")[-1]) for p in keep} == KEEP
    latest = json.loads((directory / "latest.json").read_text())
    assert (ROOT / latest["checkpoint"]).resolve() in {p.resolve() for p in keep}
    evaluation = OUT / "evaluation/sweep_5k_10k_20k_80862/manifest.json"
    if evaluation.exists():
        for entry in json.loads(evaluation.read_text())["checkpoints"].values():
            p = OUT / entry["path"]
            if p.parent == directory:
                assert p in keep, "Checkpoint required by evaluation must be retained"
    original_hashes = {p.name: file_hash(p) for p in keep}
    untouched = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in directory.iterdir()
        if p.is_file() and p.suffix != ".pt"
    }
    report = {
        "status": "planned",
        "directory": str(directory.relative_to(ROOT)),
        "kept": original_hashes,
        "removed": [
            {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in remove
        ],
        "removed_bytes": sum(p.stat().st_size for p in remove),
        "retained_bytes": sum(p.stat().st_size for p in keep),
        "scope": "Only numeric ckpt_*.pt in completed joint_6h_from5000; original 5k run, other experiments, datasets, logs, metrics, figures and evaluation files untouched.",
    }
    if args.apply:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        audit = OUT / f"checkpoint_retention_{stamp}.json"
        save_json(audit, report)
        for path in remove:
            path.unlink()
        assert sorted(directory.glob("ckpt_*.pt")) == keep
        assert {p.name: file_hash(p) for p in keep} == original_hashes
        assert all(
            (directory / name).stat().st_size == meta[0]
            and (directory / name).stat().st_mtime_ns == meta[1]
            for name, meta in untouched.items()
        )
        report.update(status="completed", audit_file=str(audit.relative_to(ROOT)))
        save_json(audit, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "kept_count": len(keep),
                "deleted_count": len(remove) if args.apply else 0,
                "freed_gib": report["removed_bytes"] / 2**30 if args.apply else 0,
                "retained_gib": report["retained_bytes"] / 2**30,
                "audit_file": report.get("audit_file"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
