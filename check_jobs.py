"""
Check HPC job status after a SLURM array run.

Scans logs/ for .err files and results/ for missing task outputs,
then prints a summary of which tasks failed and what the errors were.

Usage:
    python check_jobs.py
    python check_jobs.py --logs_dir logs --results_dir results/parameter_sweep
    python check_jobs.py --job_id 12345678   # filter to a specific SLURM job
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Check HPC job errors from SLURM logs")
    parser.add_argument("--logs_dir", default="logs", help="Directory containing SLURM .out/.err files")
    parser.add_argument("--results_dir", default="results/parameter_sweep", help="Directory with task CSV outputs")
    parser.add_argument("--job_id", default=None, help="Filter to a specific SLURM array job ID (optional)")
    parser.add_argument("--show_ok", action="store_true", help="Also list tasks that completed successfully")
    return parser.parse_args()


def collect_log_files(logs_dir: Path, job_id: str | None):
    """Return sorted lists of .out and .err files, optionally filtered by job_id."""
    if job_id:
        out_files = sorted(logs_dir.glob(f"sweep_{job_id}_*.out"))
        err_files = sorted(logs_dir.glob(f"sweep_{job_id}_*.err"))
    else:
        out_files = sorted(logs_dir.glob("sweep_*.out"))
        err_files = sorted(logs_dir.glob("sweep_*.err"))
    return out_files, err_files


def task_id_from_filename(path: Path) -> str | None:
    """Extract array task ID from filename like sweep_12345_42.err -> '42'."""
    m = re.search(r"sweep_\d+_(\d+)\.(out|err)$", path.name)
    return m.group(1) if m else None


def check_err_file(err_path: Path) -> list[str]:
    """Return non-empty, non-whitespace lines from a .err file."""
    try:
        lines = err_path.read_text(errors="replace").splitlines()
        return [l for l in lines if l.strip()]
    except OSError:
        return [f"<could not read {err_path}>"]


def check_out_file(out_path: Path) -> dict:
    """Parse .out file for start/finish markers and inline 'Error with config' lines."""
    info = {"started": False, "finished": False, "inline_errors": []}
    try:
        for line in out_path.read_text(errors="replace").splitlines():
            if line.startswith("Starting task"):
                info["started"] = True
            elif line.startswith("Finished task"):
                info["finished"] = True
            elif line.startswith("Error with config"):
                info["inline_errors"].append(line.strip())
    except OSError:
        pass
    return info


def find_missing_tasks(results_dir: Path, n_tasks: int) -> list[int]:
    """Return task IDs for which no output CSV exists."""
    existing = {int(m.group(1)) for p in results_dir.glob("task_*.csv")
                if (m := re.search(r"task_(\d+)\.csv$", p.name))}
    return [i for i in range(n_tasks) if i not in existing]


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    results_dir = Path(args.results_dir)

    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return

    out_files, err_files = collect_log_files(logs_dir, args.job_id)

    if not out_files and not err_files:
        label = f" for job {args.job_id}" if args.job_id else ""
        print(f"No sweep log files found in '{logs_dir}'{label}.")
        return

    # Map task_id -> paths
    out_map = {task_id_from_filename(p): p for p in out_files}
    err_map = {task_id_from_filename(p): p for p in err_files}
    all_task_ids = sorted(set(out_map) | set(err_map), key=lambda x: int(x) if x else -1)

    failed = []       # (task_id, reason_lines)
    incomplete = []   # started but never finished
    ok = []

    for tid in all_task_ids:
        if tid is None:
            continue

        out_info = check_out_file(out_map[tid]) if tid in out_map else {"started": False, "finished": False, "inline_errors": []}
        stderr_lines = check_err_file(err_map[tid]) if tid in err_map else []

        reasons = []
        if stderr_lines:
            reasons.extend(stderr_lines)
        if out_info["inline_errors"]:
            reasons.extend(out_info["inline_errors"])

        if reasons:
            failed.append((tid, reasons))
        elif out_info["started"] and not out_info["finished"]:
            incomplete.append(tid)
        else:
            ok.append(tid)

    # Missing output CSVs (tasks that produced no result file)
    missing_csvs: list[int] = []
    if results_dir.exists():
        n_tasks = len(all_task_ids)
        missing_csvs = find_missing_tasks(results_dir, n_tasks)
        # Remove tasks already flagged as failed/incomplete
        flagged = {int(t) for t, _ in failed} | {int(t) for t in incomplete}
        missing_csvs = [t for t in missing_csvs if t not in flagged]

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    total = len(all_task_ids)
    print(f"\n{'='*60}")
    print(f"  JOB CHECK SUMMARY  ({total} tasks found in logs)")
    print(f"{'='*60}")
    print(f"  OK / completed  : {len(ok)}")
    print(f"  Failed (errors) : {len(failed)}")
    print(f"  Incomplete      : {len(incomplete)}  (started, never finished)")
    print(f"  Missing CSV out : {len(missing_csvs)}  (no task_XXXX.csv produced)")
    print(f"{'='*60}\n")

    if failed:
        print(f"FAILED TASKS ({len(failed)})")
        print("-" * 60)
        # Group by unique error messages to avoid repeating the same wall of text
        error_groups: dict[str, list[str]] = defaultdict(list)
        for tid, reasons in failed:
            key = "\n".join(reasons[:10])   # first 10 lines as grouping key
            error_groups[key].append(tid)

        for error_text, tids in error_groups.items():
            print(f"Tasks: {', '.join(tids)}")
            for line in error_text.splitlines():
                print(f"  {line}")
            print()

    if incomplete:
        print(f"INCOMPLETE TASKS (started but no 'Finished' line): {', '.join(incomplete)}\n")

    if missing_csvs:
        print(f"TASKS WITH NO OUTPUT CSV: {missing_csvs}\n")

    if args.show_ok and ok:
        print(f"OK TASKS: {', '.join(ok)}\n")

    if not failed and not incomplete and not missing_csvs:
        print("All tasks completed successfully.")


if __name__ == "__main__":
    main()
