#!/usr/bin/env python3
"""Aggregate completed trial outcomes from metrics JSON or inspection JSONL.

Examples:
  python scripts/summarize_runs.py 'outputs/runs/*/metrics.json' --out outputs/summary.json
  python scripts/summarize_runs.py outputs/run/inspections.jsonl

Use either metrics files or JSONL logs for the same run, not both: individual
records without a run_id/trial pair cannot safely be deduplicated.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sorter.measurement import atomic_write_json, summarize_records


def load_records(path: Path) -> tuple[list[dict], dict]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{index}: invalid JSONL") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path.name}:{index}: expected an object")
                records.append(record)
        return records, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: expected an object or array")
    for key in ("trials", "cycles", "records"):
        if isinstance(payload.get(key), list):
            return payload[key], {key: payload.get(key) for key in ("status", "stop_reason") if key in payload}
    if "success" in payload:
        return [payload], {}
    # Startup failures can contain no trial rows; retain this run's status.
    return [], {key: payload.get(key) for key in ("status", "stop_reason", "error") if key in payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Metrics JSON/JSONL paths or glob patterns")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    files = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            parser.error(f"No input matches: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path not in files:
                files.append(path)
    records, runs, seen = [], [], set()
    duplicates = 0
    for path in files:
        try:
            rows, metadata = load_records(path)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        for row in rows:
            if not isinstance(row, dict):
                parser.error(f"{path.name}: each trial must be an object")
            identity = (str(row.get("run_id")), str(row.get("trial")))
            if row.get("run_id") is not None and row.get("trial") is not None:
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
            records.append(row)
        runs.append({"file": path.name, "record_count": len(rows), **metadata})
    result = {"summary": summarize_records(records), "runs": runs, "duplicate_records_skipped": duplicates}
    if args.out:
        atomic_write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
