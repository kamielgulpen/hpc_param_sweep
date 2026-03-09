"""
SLURM array task runner for the parameter sweep.

Each array task (SLURM_ARRAY_TASK_ID) maps to one (n_communities, pref_attachment)
combination. Results are written to a per-task CSV file in --output_dir.

Usage:
    python run_task.py --task_id $SLURM_ARRAY_TASK_ID --output_dir results/parameter_sweep
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from parameter_sweep import NetworkConfig, ContagionAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Run one parameter sweep task for SLURM array job")
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task index (0-based). Defaults to SLURM_ARRAY_TASK_ID env var.')
    parser.add_argument('--output_dir', type=str, default='results/parameter_sweep',
                        help='Directory to write per-task CSV results')
    args = parser.parse_args()

    # Resolve task ID from argument or environment
    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    net = NetworkConfig()
    combinations = net.all_combinations()
    n_total = len(combinations)
    print(n_total)

    if task_id >= n_total:
        print(f"Task {task_id} out of range (only {n_total} combinations). Exiting.")
        return

    n_comms, pref_att = combinations[task_id]
    print(f"Task {task_id}/{n_total - 1}: n_communities={n_comms:.4f}, pref_attachment={pref_att:.4f}")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    out_file = out_path / f"task_{task_id:04d}.csv"
    if out_file.exists():
        print(f"Already complete: {out_file}. Skipping.")
        return

    analyzer = ContagionAnalyzer()
    print(n_comms, pref_att)
    result = analyzer.run_single(n_comms, pref_att)

    if result:
        df = pd.DataFrame(result)
        df.to_csv(out_file, index=False)
        print(f"Saved {len(df)} rows to {out_file}")
    else:
        print(f"No results for task {task_id} (folder missing or error)")


if __name__ == '__main__':
    main()
