#!/bin/bash
# =============================================================================
# SLURM Array Job: Parameter Sweep for Contagion Model
#
# Parameter space: 10 n_communities × 10 pref_attachment = 100 combinations
# Each array task runs one (n_communities, pref_attachment) combination.
#
# Workflow:
#   1. sbatch submit_warmup.sh          # pre-compile numba kernel (run once)
#   2. sbatch --dependency=afterok:<warmup_job_id> submit_job.sh
#   3. python merge_results.py          # after all tasks finish
# =============================================================================

#SBATCH --job-name=param_sweep
#SBATCH --array=0-10               # 100 tasks (0-indexed), one per parameter combo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # numba parallel=True uses threads
#SBATCH --time=02:00:00            # adjust per your cluster limits
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# --- Environment setup -------------------------------------------------------
# Adjust the lines below to match your cluster's module system and venv path.

# module load python/3.11          # uncomment and set Python version if needed
# source /path/to/your/venv/bin/activate   # uncomment and set your venv path

# Alternatively, if using conda:
# module load anaconda
# conda activate your_env_name

# -----------------------------------------------------------------------------

mkdir -p logs results/parameter_sweep

echo "Starting task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"

python run_task.py \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --output_dir results/parameter_sweep

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"
