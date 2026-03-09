#!/bin/bash
# =============================================================================
# Warmup job: pre-compiles the numba kernel into the __pycache__ on the
# compute nodes before the array job starts. Run this once first, then submit
# the array job with a dependency on this job ID.
#
#   warmup_id=$(sbatch --parsable submit_warmup.sh)
#   sbatch --dependency=afterok:${warmup_id} submit_job.sh
# =============================================================================

#SBATCH --job-name=numba_warmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/warmup_%j.out
#SBATCH --error=logs/warmup_%j.err

# --- Environment setup (mirror submit_job.sh) --------------------------------
# module load python/3.11
# source /path/to/your/venv/bin/activate

# -----------------------------------------------------------------------------

mkdir -p logs

echo "Pre-compiling numba kernel at $(date)"
python -c "
from parameter_sweep import _complex_contagion_kernel, ContagionSimulator
import networkx as nx, numpy as np
# Trigger JIT compilation with a tiny dummy graph
G = nx.path_graph(10)
sim = ContagionSimulator(G)
sim.complex_contagion(threshold=0.1, threshold_type='fractional',
                      initial_infected=1, max_steps=2, n_simulations=1)
print('Numba kernel compiled and cached.')
"
echo "Warmup done at $(date)"
