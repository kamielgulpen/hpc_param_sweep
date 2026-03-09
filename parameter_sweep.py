"""
Memory-Optimized Contagion Parameter Sweep.

Sweeps across (n_communities, pref_attachment) network parameter combinations,
running complex contagion simulations with incremental checkpointing.
"""

import gc
import pickle
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numba
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# Contagion kernel (JIT-compiled)
# =============================================================================

@numba.njit(parallel=True, cache=True)
def _complex_contagion_kernel(data, indices, indptr, degree, state, threshold, is_fractional, max_steps):
    """
    JIT-compiled contagion kernel. Runs all simulations in parallel over nodes.

    Args:
        data, indices, indptr: CSR sparse adjacency matrix components
        degree: (n,) node degree array
        state: (n, n_sims) initial infection state — modified in-place
        threshold: adoption threshold value
        is_fractional: True for fractional threshold, False for absolute
        max_steps: maximum simulation steps

    Returns:
        time_series: (actual_steps+1, n_sims) int64 array of infected counts
    """
    n, n_sims = state.shape
    infected_counts = np.empty((n, n_sims), dtype=np.float64)
    time_series = np.empty((max_steps + 1, n_sims), dtype=np.int64)

    for s in range(n_sims):
        t = np.int64(0)
        for i in range(n):
            t += np.int64(state[i, s] > 0.0)
        time_series[0, s] = t

    actual_steps = 0

    for step in range(max_steps):
        for i in numba.prange(n):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            for s in range(n_sims):
                val = 0.0
                for j_ptr in range(row_start, row_end):
                    val += data[j_ptr] * state[indices[j_ptr], s]
                infected_counts[i, s] = val

        for i in numba.prange(n):
            d = degree[i]
            for s in range(n_sims):
                if state[i, s] == 0.0:
                    ic = infected_counts[i, s]
                    if is_fractional:
                        meets = (d > 0.0) and (ic / d >= threshold)
                    else:
                        meets = ic >= threshold
                    if meets:
                        state[i, s] = 1.0

        all_done = True
        for s in range(n_sims):
            t = np.int64(0)
            for i in range(n):
                t += np.int64(state[i, s] > 0.0)
            time_series[step + 1, s] = t
            if t != np.int64(n) and t != time_series[step, s]:
                all_done = False

        actual_steps += 1
        if all_done:
            break

    return time_series[:actual_steps + 1]


# =============================================================================
# Network simulation
# =============================================================================

class ContagionSimulator:
    """Simulates complex contagion spreading on networks."""

    def __init__(self, network, name="Network"):
        self.G = network
        self.name = name
        self.n = len(network)
        adj = nx.to_scipy_sparse_array(network, format='csr', dtype=np.float64)
        self.adj = adj
        self.degree = np.array(self.adj.sum(axis=1)).flatten()

    def _seed_state(self, state, n_simulations, seeding, initial_infected):
        """Initialize infection state based on seeding strategy."""
        if isinstance(seeding, np.ndarray):
            for sim in range(n_simulations):
                nodes = np.random.choice(seeding, initial_infected, replace=False)
                state[nodes, sim] = 1.0
        elif seeding == 'focal_neighbors':
            for sim in range(n_simulations):
                focal = np.random.randint(self.n)
                state[focal, sim] = 1.0
                neighbors = self.adj.indices[self.adj.indptr[focal]:self.adj.indptr[focal + 1]]
                state[neighbors, sim] = 1.0
        else:  # 'random'
            for sim in range(n_simulations):
                nodes = np.random.choice(self.n, initial_infected, replace=False)
                state[nodes, sim] = 1.0

    def complex_contagion(self, threshold=2, threshold_type='absolute',
                         initial_infected=1, max_steps=30, n_simulations=1,
                         seeding='random'):
        """
        Deterministic threshold model using JIT-compiled kernel.

        Args:
            threshold: Absolute count or fraction of neighbors needed
            threshold_type: 'absolute' or 'fractional'
            initial_infected: Number of seed nodes
            max_steps: Maximum simulation steps
            n_simulations: Number of parallel runs
            seeding: 'random', 'focal_neighbors', or np.ndarray of node indices
        """
        state = np.zeros((self.n, n_simulations), dtype=np.float64)
        self._seed_state(state, n_simulations, seeding, initial_infected)

        is_fractional = (threshold_type != 'absolute')
        ts = _complex_contagion_kernel(
            self.adj.data, self.adj.indices, self.adj.indptr,
            self.degree, state, float(threshold), is_fractional, max_steps
        )
        return [[int(ts[t, sim]) for t in range(ts.shape[0])]
                for sim in range(n_simulations)]


def load_networks(folder: str) -> dict:
    """Load all .pkl network objects from a folder (skips non-graph objects)."""
    networks = {}
    for pkl_file in sorted(Path(folder).glob('*.pkl')):
        name = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, nx.Graph):
            if not isinstance(obj.graph, nx.Graph):
                continue
            networks[name] = obj
        else:
            networks[name] = obj
    return networks


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """Simulation parameters."""
    n_simulations: int = 20
    max_steps: int = 50
    threshold_type: str = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 0.30
    n_thresholds: int = 4

    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)


@dataclass
class NetworkConfig:
    """Network parameters."""
    base_folder: str = "Data/networks/werkschool"
    scale: float = 0.1
    reciprocity: int = 1
    transitivity: int = 0
    bridge: float = 0.2
    n_communities_range: Tuple[float, float, int] = (1, 1000, 10)
    preferential_attachment_range: Tuple[float, float, int] = (0, 0.99, 10)

    def folder_path(self, n_comms: float, pref_att: float) -> str:
        return (f"{self.base_folder}/scale={self.scale}_comms={n_comms}_"
                f"recip={self.reciprocity}_trans={self.transitivity}_"
                f"pa={pref_att}_bridge={self.bridge}")

    @property
    def community_values(self) -> np.ndarray:
        return np.linspace(*self.n_communities_range)

    @property
    def attachment_values(self) -> np.ndarray:
        return np.linspace(*self.preferential_attachment_range)

    def all_combinations(self) -> List[Tuple[float, float]]:
        return list(product(self.community_values, self.attachment_values))


# =============================================================================
# Memory management
# =============================================================================

class MemoryManager:

    @staticmethod
    def clear_memory():
        gc.collect()

    @staticmethod
    def save_checkpoint(data: pd.DataFrame, filepath: Path, append: bool = True):
        if append and filepath.exists():
            data.to_csv(filepath, mode='a', header=False, index=False)
        else:
            data.to_csv(filepath, index=False)

    @staticmethod
    def load_checkpoint(filepath: Path) -> pd.DataFrame:
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()

    @staticmethod
    def get_completed_configs(filepath: Path) -> set:
        if not filepath.exists():
            return set()
        df = pd.read_csv(filepath)
        if 'n_communities' in df.columns and 'pref_attachment' in df.columns:
            return set(zip(df['n_communities'], df['pref_attachment']))
        return set()


# =============================================================================
# Sweep runner
# =============================================================================

class ContagionAnalyzer:
    """Run parameter sweep with memory-efficient batching and checkpointing."""

    def __init__(self, sim_config: SimulationConfig = None, net_config: NetworkConfig = None):
        self.sim = sim_config or SimulationConfig()
        self.net = net_config or NetworkConfig()
        self.mm = MemoryManager()

    def run_parameter_sweep(self, n_iterations: int = 100, shuffle: bool = True,
                            batch_size: int = 5, output_dir: str = "results/parameter_sweep",
                            resume: bool = True) -> pd.DataFrame:
        """Run parameter sweep with batching and checkpointing."""
        print("\n" + "="*70)
        print("MEMORY-OPTIMIZED CONTAGION PARAMETER SWEEP")
        print("="*70)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = out_path / "checkpoint_sweep.csv"

        params = self.net.all_combinations()
        if shuffle:
            np.random.shuffle(params)
        params = params[:n_iterations]

        completed = self.mm.get_completed_configs(checkpoint_file) if resume else set()
        print(f"Found {len(completed)} completed configurations")
        params = [(nc, pa) for nc, pa in params if (nc, pa) not in completed]
        print(f"Processing {len(params)} remaining configurations")

        batch_results = []
        for idx, (n_comms, pref_att) in enumerate(tqdm(params, desc="Parameter sweep")):
            result = self.run_single(n_comms, pref_att)
            if result:
                batch_results.extend(result)

            if (idx + 1) % batch_size == 0 and batch_results:
                self.mm.save_checkpoint(pd.DataFrame(batch_results), checkpoint_file, append=True)
                print(f"\nCheckpoint saved: {len(batch_results)} results")
                batch_results = []
                self.mm.clear_memory()

        if batch_results:
            self.mm.save_checkpoint(pd.DataFrame(batch_results), checkpoint_file, append=True)

        return self.mm.load_checkpoint(checkpoint_file)

    def run_single(self, n_comms: float, pref_att: float) -> Optional[List[Dict]]:
        """Run simulation for one (n_communities, pref_attachment) combination."""
        folder = Path(self.net.folder_path(n_comms, pref_att))
        if not folder.exists():
            return None

        try:
            network_graphs = load_networks(str(folder))
            target_groups = {"geslacht", "etngrp_geslacht_lft_oplniv",
                             "geslacht_oplniv", "lft", "etngrp_oplniv"}
            networks = {k: v.graph for k, v in network_graphs.items() if k in target_groups}

            contested_results, ratios = self._sweep_thresholds(networks)

            results = []
            for net_name, thresh_results in contested_results.items():
                if net_name not in ratios:
                    continue
                for thresh_idx, stat_val in thresh_results.items():
                    results.append({
                        'n_communities': n_comms,
                        'pref_attachment': pref_att,
                        'network': net_name,
                        'threshold_idx': thresh_idx,
                        'threshold_value': self.sim.thresholds[thresh_idx],
                        'median_final_adoption': stat_val["mean"],
                        'internal_variance_adoption': stat_val["variance"],
                        'ratio': ratios[net_name]
                    })

            del network_graphs, networks
            self.mm.clear_memory()
            return results

        except Exception as e:
            print(f"Error with config ({n_comms}, {pref_att}): {e}")
            return None

    def _sweep_thresholds(self, networks: Dict) -> Tuple[Dict, Dict]:
        """Sweep contagion thresholds across all networks."""
        results, ratios = {}, {}

        for name, G in networks.items():
            try:
                df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")
                ratios[name] = df_n.n.max() / df_n.n.sum()
                del df_n
            except FileNotFoundError:
                continue

            sim = ContagionSimulator(G, name)
            initial = int(len(G) * self.sim.initial_infected_fraction)

            finals = {}
            for i, tau in enumerate(self.sim.thresholds):
                ts_list = sim.complex_contagion(
                    threshold=tau, threshold_type=self.sim.threshold_type,
                    seeding='focal_neighbors', max_steps=self.sim.max_steps,
                    n_simulations=self.sim.n_simulations, initial_infected=initial
                )
                finals[i] = {
                    "mean": np.mean([ts[-1] for ts in ts_list]),
                    "variance": np.var([ts[-1] for ts in ts_list])
                }
                del ts_list

            results[name] = finals
            del sim
            gc.collect()

        return results, ratios


# =============================================================================
# Visualization
# =============================================================================

class Visualizer:

    @staticmethod
    def plot_variance_vs_ratio(df: pd.DataFrame, out: str = "variance_analysis.png"):
        """Plot adoption variance vs group size ratio across thresholds."""
        plot_data = []
        for net in df['network'].unique():
            for thresh in df['threshold_value'].unique():
                subset = df[(df['network'] == net) & (df['threshold_value'] == thresh)]
                if len(subset) > 1:
                    plot_data.append({
                        'network': net,
                        'threshold': thresh,
                        'variance': subset['median_final_adoption'].var(),
                        'ratio': subset['ratio'].iloc[0] if 'ratio' in subset.columns else 0
                    })

        if not plot_data:
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(data=pd.DataFrame(plot_data), x='ratio', y='variance',
                        hue='threshold', palette='viridis', s=150, alpha=0.7, ax=ax)
        ax.set_xlabel('Group Size Ratio (max/total)', fontsize=13)
        ax.set_ylabel('Adoption Variance', fontsize=13)
        ax.set_title('Contagion Variance vs Network Homogeneity', fontsize=15, fontweight='bold')
        ax.legend(title='Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()
        print(f"Saved: {out}")

    @staticmethod
    def plot_threshold_curves(df: pd.DataFrame, out: str = "threshold_curves.png"):
        """Plot mean adoption curves across threshold values per network."""
        fig, ax = plt.subplots(figsize=(12, 7))
        for net in df['network'].unique():
            grouped = (df[df['network'] == net]
                       .groupby('threshold_value')['median_final_adoption'].mean())
            ax.plot(grouped.index, grouped.values, marker='o',
                    label=net, linewidth=2, markersize=5, alpha=0.7)
        ax.set_xlabel('Adoption Threshold', fontsize=13)
        ax.set_ylabel('Final Adoption', fontsize=13)
        ax.set_title('Threshold Sensitivity Analysis', fontsize=15, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()
        print(f"Saved: {out}")


# =============================================================================
# Entry point
# =============================================================================

def run_parameter_sweep(n_iterations: int = 100, batch_size: int = 5,
                        output_dir: str = "results/parameter_sweep",
                        resume: bool = True) -> pd.DataFrame:
    analyzer = ContagionAnalyzer()
    results = analyzer.run_parameter_sweep(
        n_iterations=n_iterations,
        batch_size=batch_size,
        output_dir=output_dir,
        resume=resume
    )

    out_path = Path(output_dir)
    results.to_csv(out_path / "sweep_results_final.csv", index=False)
    print(f"\nFinal results: {out_path / 'sweep_results_final.csv'}")

    # viz = Visualizer()
    # sample = results.sample(min(10000, len(results))) if len(results) > 10000 else results
    # viz.plot_variance_vs_ratio(sample, str(out_path / "variance_analysis.png"))
    # viz.plot_threshold_curves(sample, str(out_path / "threshold_curves.png"))

    return results


if __name__ == "__main__":
    np.random.seed(42)
    results = run_parameter_sweep(n_iterations=100, batch_size=1, resume=True)
    print("\nSummary:")
    print(results.groupby('threshold_value')['median_final_adoption'].describe())
