"""
Merge per-task CSV files into a single result file and generate visualizations.

Run after all SLURM array tasks have completed:
    python merge_results.py --input_dir results/parameter_sweep
"""

import argparse
import pandas as pd
from pathlib import Path

from parameter_sweep import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Merge per-task sweep results")
    parser.add_argument('--input_dir', type=str, default='results/parameter_sweep',
                        help='Directory containing task_XXXX.csv files')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV path (default: <input_dir>/sweep_results_final.csv)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating visualizations')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file) if args.output_file else input_dir / 'sweep_results_final.csv'

    task_files = sorted(input_dir.glob('task_*.csv'))
    if not task_files:
        print(f"No task_*.csv files found in {input_dir}")
        return

    print(f"Merging {len(task_files)} task files...")
    df = pd.concat([pd.read_csv(f) for f in task_files], ignore_index=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")

    if not args.no_plots:
        viz = Visualizer()
        sample = df.sample(min(10000, len(df))) if len(df) > 10000 else df
        viz.plot_variance_vs_ratio(sample, str(input_dir / "variance_analysis.png"))
        viz.plot_threshold_curves(sample, str(input_dir / "threshold_curves.png"))


if __name__ == '__main__':
    main()
