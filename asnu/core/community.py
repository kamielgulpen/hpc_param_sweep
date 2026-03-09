"""
Community detection and management module for ASNU.

This module provides functions for creating, populating, and analyzing
community structures within population networks. It implements agent-based
decision making for community assignment and provides various community
size distributions.

Functions
---------
build_group_pair_to_communities_lookup : Create lookup for group pairs to communities
populate_communities : Assign nodes to communities using group-aligned decision making
find_separated_groups : Identify groups with minimal inter-connections to seed communities
analyze_community_distribution : Analyze distribution of communities and groups
connect_all_within_communities : Create fully connected subgraphs within communities
fill_unfulfilled_group_pairs : Complete group pairs that didn't reach target edge count
export_community_edge_distribution : Export edge distribution to CSV file
export_community_node_distribution : Export node distribution to CSV file
"""
import csv
import random
from collections import Counter
from itertools import product

import numpy as np


def build_group_pair_to_communities_lookup(G, verbose=False):
    """
    Create a lookup dictionary mapping each group pair to their shared communities.
    The algorithm will also resamble the structure when some groups are more representative
    in communities than others.

    This precomputes which communities contain which group pairs, making link
    creation much faster by avoiding repeated community membership checks.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community information
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Mapping from (src_id, dst_id) to list of shared community IDs
    """
    if verbose:
        print("Building community lookup for group pairs...")

    group_pair_to_communities = {}

    for community_id in range(G.number_of_communities):
        groups_in_community = G.communities_to_groups.get(community_id, [])
        for src_id in groups_in_community:
            for dst_id in groups_in_community:
                pair_key = (src_id, dst_id)

                if pair_key not in group_pair_to_communities:
                    group_pair_to_communities[pair_key] = []

                group_pair_to_communities[pair_key].append(community_id)

    if verbose:
        avg_communities = np.mean([len(v) for v in group_pair_to_communities.values()])
        print(f"  Found {len(group_pair_to_communities)} group pairs")
        print(f"  Average communities per pair: {avg_communities:.1f}")

    return group_pair_to_communities


def _process_nodes_python(G, all_nodes, node_groups, community_composition,
                          community_sizes, group_exposure, ideal,
                          target_counts, total_nodes):
    """Pure-Python fallback for the node assignment loop."""
    for node_idx in range(len(all_nodes)):
        node = all_nodes[node_idx]
        group = node_groups[node_idx]

        current_exp = group_exposure[group, :]

        hypothetical_exp = current_exp + community_composition
        hypothetical_totals = hypothetical_exp.sum(axis=1, keepdims=True)
        hypothetical_totals = np.maximum(hypothetical_totals, 1e-10)
        hypothetical_dist = hypothetical_exp / hypothetical_totals

        diff = hypothetical_dist - ideal[group, :]
        distances = np.sqrt((diff ** 2).sum(axis=1))

        if target_counts is not None:
            full_mask = community_sizes >= target_counts
            distances[full_mask] = np.inf

        temperature = 1.0 - (node_idx / total_nodes)
        if temperature > 0.05:
            valid_mask = distances < np.inf
            if valid_mask.sum() > 1:
                d = distances[valid_mask]
                scaled = -d / (temperature + 1e-10)
                scaled = scaled - scaled.max()
                probs = np.exp(scaled)
                probs = probs / probs.sum()
                valid_indices = np.where(valid_mask)[0]
                best_community = np.random.choice(valid_indices, p=probs)
            else:
                best_community = np.argmin(distances)
        else:
            best_community = np.argmin(distances)

        G.communities_to_nodes[(best_community, group)].append(node)
        G.nodes_to_communities[node] = best_community
        G.communities_to_groups[best_community].append(group)

        group_exposure[group, :] += community_composition[best_community, :]
        mask = community_composition[best_community, :] > 0
        group_exposure[mask, group] += 1

        community_composition[best_community, group] += 1
        community_sizes[best_community] += 1

        if (node_idx + 1) % 500 == 0:
            print(f"Assigned {node_idx + 1}/{total_nodes} nodes ({100*(node_idx+1)/total_nodes:.1f}%)")


def populate_communities(G, num_communities, community_size_distribution='natural'):
    """
    Assign nodes to communities using group-aligned agent decision making.

    Each node (agent) chooses a community by optimizing for its GROUP's collective
    alignment with the ideal link distribution. Uses vectorized operations for speed.

    Agent Decision-Making Protocol:
    1. Track Current Distribution: Group exposure to other groups across communities
    2. Calculate Distance from Ideal: Compare against ideal link distribution
    3. Evaluate Action Consequences: Vectorized distance calculation for all communities
    4. Select Optimal Action: Choose community minimizing distance to ideal

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Number of communities to create
    seed_groups : list
        Initial group IDs to seed each community
    community_size_distribution : str or array-like, optional
        Controls community size distribution
    """
    total_nodes = len(list(G.graph.nodes))
    n_groups = int(len(G.group_ids))

    # Create affinity matrix from link counts
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G.maximum_num_links.items():
        affinity[i, j] = count

    # Normalize affinity to get probability matrix
    epsilon = 1e-5
    normalized = affinity / (affinity.sum(axis=1, keepdims=True) + epsilon)
    normalized[normalized == 0] = epsilon

    # Store for later use in community assignment
    G.probability_matrix = normalized.copy()
    G.number_of_communities = num_communities


    # Calculate target community sizes
    if isinstance(community_size_distribution, (list, np.ndarray)):
        target_sizes = np.array(community_size_distribution)
        if not np.isclose(target_sizes.sum(), 1.0):
            raise ValueError("Custom community_size_distribution must sum to 1")
    elif community_size_distribution == 'powerlaw':
        ranks = np.arange(1, num_communities + 1)
        sizes = 1.0 / (ranks ** 1)
        target_sizes = sizes / sizes.sum()
    elif community_size_distribution == 'uniform':
        target_sizes = np.ones(num_communities) / num_communities
    else:  # 'natural'
        target_sizes = None

    # Initialize community structures
    for community_idx in range(num_communities):
        for group_id in range(n_groups):
            G.communities_to_nodes[(community_idx, group_id)] = []
        G.communities_to_groups[community_idx] = []

    # Vectorized tracking arrays
    # community_composition[c, g] = count of group g in community c
    community_composition = np.zeros((num_communities, n_groups), dtype=np.float64)
    community_sizes = np.zeros(num_communities, dtype=np.int32)

    # group_exposure[g, h] = cumulative exposure of group g to group h
    group_exposure = np.zeros((n_groups, n_groups), dtype=np.float64)

    # Ideal distribution from link probability matrix
    ideal = G.probability_matrix.copy()  # shape: (n_groups, n_groups)

    # Target sizes for distribution control
    if target_sizes is not None:
        target_counts = (target_sizes * total_nodes).astype(np.int32)
        remainder = total_nodes - target_counts.sum()
        for i in range(remainder):
            target_counts[i % num_communities] += 1

    # Shuffle and get node groups as array
    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    # Process nodes — try Rust, fall back to Python


    try:
        from asnu_rust import process_nodes
        print("Using Rust backend for node processing...")
        assignments = process_nodes(
            all_nodes.astype(np.int64), node_groups.astype(np.int64),
            community_composition, community_sizes,
            group_exposure, ideal,
            target_counts if target_sizes is not None else None,
            total_nodes,
        )
        for i in range(len(all_nodes)):
            node = int(all_nodes[i])
            comm = int(assignments[i])
            group = int(node_groups[i])
            G.communities_to_nodes[(comm, group)].append(node)
            G.nodes_to_communities[node] = comm
            G.communities_to_groups[comm].append(group)
    except ImportError:
        _process_nodes_python(
            G, all_nodes, node_groups, community_composition,
            community_sizes, group_exposure, ideal,
            target_counts if target_sizes is not None else None,
            total_nodes,
        )

    print(f"\nCommunity population complete: {len(all_nodes)} nodes assigned")
    # === DIAGNOSTIC 1: Community Size Distribution ===
    print("\n" + "="*60)
    print("DIAGNOSTIC 1: Community Size Distribution")
    print("="*60)
    non_empty = community_sizes[community_sizes > 0]
    print(f"Total communities: {num_communities}")
    print(f"Non-empty communities: {len(non_empty)}")
    print(f"Empty communities: {num_communities - len(non_empty)}")
    print(f"Size stats - Mean: {np.mean(non_empty):.1f}, Std: {np.std(non_empty):.1f}")
    print(f"Size stats - Min: {np.min(non_empty)}, Max: {np.max(non_empty)}, Median: {np.median(non_empty):.1f}")

    # Size distribution buckets
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    print(f"Size percentiles: {dict(zip(percentiles, [int(np.percentile(non_empty, p)) for p in percentiles]))}")

    # Top 10 largest communities
    top_indices = np.argsort(community_sizes)[-10:][::-1]
    print(f"Top 10 largest communities: {[(int(i), int(community_sizes[i])) for i in top_indices]}")

    # === DIAGNOSTIC 2: Group Distribution Across Communities ===
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Group Distribution Across Communities")
    print("="*60)

    # For each group, count how many communities it appears in
    groups_per_community = (community_composition > 0).sum(axis=1)  # groups per community
    communities_per_group = (community_composition > 0).sum(axis=0)  # communities per group

    print(f"Groups per community - Mean: {np.mean(groups_per_community):.1f}, Std: {np.std(groups_per_community):.1f}")
    print(f"Communities per group - Mean: {np.mean(communities_per_group):.1f}, Std: {np.std(communities_per_group):.1f}")

    # Show group concentration in top communities
    print("\nGroup presence in top 5 largest communities:")
    for comm_idx in top_indices[:5]:
        groups_present = np.where(community_composition[comm_idx, :] > 0)[0]
        group_counts = community_composition[comm_idx, groups_present].astype(int)
        # Sort by count
        sorted_idx = np.argsort(group_counts)[::-1]
        top_groups = [(int(groups_present[i]), int(group_counts[i])) for i in sorted_idx[:5]]
        print(f"  Community {comm_idx} (size {int(community_sizes[comm_idx])}): top groups {top_groups}")

    # === DIAGNOSTIC 3: Exposure vs Ideal Alignment ===
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: Group Exposure vs Ideal Alignment")
    print("="*60)

    # Normalize group exposure to get actual distribution
    group_exposure_totals = group_exposure.sum(axis=1, keepdims=True)
    group_exposure_totals = np.maximum(group_exposure_totals, 1e-10)
    actual_distribution = group_exposure / group_exposure_totals

    # Calculate per-group distance from ideal
    group_distances = np.sqrt(((actual_distribution - ideal) ** 2).sum(axis=1))

    print(f"Distance from ideal - Mean: {np.mean(group_distances):.4f}, Std: {np.std(group_distances):.4f}")
    print(f"Distance from ideal - Min: {np.min(group_distances):.4f}, Max: {np.max(group_distances):.4f}")

    # Best and worst aligned groups
    best_groups = np.argsort(group_distances)[:5]
    worst_groups = np.argsort(group_distances)[-5:][::-1]
    print(f"\nBest aligned groups (lowest distance): {[(int(g), f'{group_distances[g]:.4f}') for g in best_groups]}")
    print(f"Worst aligned groups (highest distance): {[(int(g), f'{group_distances[g]:.4f}') for g in worst_groups]}")

    # Show detailed comparison for worst aligned group
    worst_group = worst_groups[0]
    print(f"\nDetailed view for worst aligned group {worst_group}:")
    print(f"  Ideal distribution (top 5):  {sorted(enumerate(ideal[worst_group]), key=lambda x: -x[1])[:5]}")
    print(f"  Actual distribution (top 5): {sorted(enumerate(actual_distribution[worst_group]), key=lambda x: -x[1])[:5]}")

    print("="*60 + "\n")


def _process_nodes_capacity_python(G, all_nodes, node_groups, num_communities,
                                   target, total_nodes, target_counts):
    """Pure-Python fallback for capacity-based node assignment using matrix ops.

    Optimized: only evaluates non-empty communities, avoids matrix copies,
    pre-allocates with room to grow.
    """
    n_groups = target.shape[0]

    # Pre-allocate with extra room for dynamic growth
    capacity = num_communities + num_communities // 20
    comp = np.zeros((capacity, n_groups), dtype=np.float64)
    community_sizes = np.zeros(capacity, dtype=np.int64)
    num_active = 0  # number of communities actually in use

    # Global accumulated edge reservations: (n_groups, n_groups)
    accumulated = np.zeros((n_groups, n_groups), dtype=np.float64)

    # Pre-compute target rows/cols for each group (avoids repeated indexing)
    target_rows = [target[g, :].copy() for g in range(n_groups)]
    target_cols = [target[:, g].copy() for g in range(n_groups)]
    
    random.shuffle(all_nodes)

    eps = 1e-6
    for node_idx in range(len(all_nodes)):
        node = all_nodes[node_idx]
        g = node_groups[node_idx]
        best_community = None

        if num_active > 0:
            # Only evaluate non-empty communities (view, no copy)
            active = comp[:num_active]

            # Hypothetical outgoing: accumulated[g,:] + comp[c,:], with 2x for self-group
            hyp_row = accumulated[g, :] + active  # new array via broadcast, no copy needed
            hyp_row[:, g] += active[:, g]         # in-place: self-group becomes 2x

            # Hypothetical incoming: accumulated[:,g] + comp[c,:], excluding self-group
            hyp_col = accumulated[:, g] + active  # new array via broadcast
            hyp_col[:, g] -= active[:, g]         # in-place: remove self-group (in hyp_row)

            # Feasibility: don't exceed budget
            tgt_row = target_rows[g]
            tgt_col = target_cols[g]
            
            feasible = np.all(hyp_row <= tgt_row * 1.01, axis=1) & np.all(hyp_col <= tgt_col * 1.01, axis=1)
            
            if target_counts is not None:
                tc_len = min(num_active, len(target_counts))
                feasible[:tc_len] &= community_sizes[:tc_len] < target_counts[:tc_len]

            if np.any(feasible):
                remaining_row = tgt_row - hyp_row
                remaining_col = tgt_col - hyp_col
                distances = np.sqrt((remaining_row ** 2).sum(axis=1) +
                                    (remaining_col ** 2).sum(axis=1))
                distances[~feasible] = np.inf

 
                # Temperature-based selection (same SA as probability mode)
                temperature = 1.0 - (node_idx / total_nodes)
                if temperature > 0.05:
                    valid_mask = feasible
                    n_valid = valid_mask.sum()
                    if n_valid > 1:
                        d = distances[valid_mask]
                        scaled = -d / (temperature + 1e-10)
                        scaled -= scaled.max()
                        probs = np.exp(scaled)
                        probs /= probs.sum()
                        valid_indices = np.where(valid_mask)[0]
                        best_community = np.random.choice(valid_indices, p=probs)
                    else:
                        best_community = np.where(valid_mask)[0][0]
                else:
                    best_community = np.argmin(distances)

        # If no feasible community, use a new empty one
        if best_community is None:
            if num_active >= comp.shape[0]:
                # Grow arrays (rare)
                extra = max(500, comp.shape[0] // 2)
                comp = np.vstack([comp, np.zeros((extra, n_groups), dtype=np.float64)])
                community_sizes = np.append(community_sizes, np.zeros(extra, dtype=np.int64))
            best_community = num_active
            num_active += 1

        # Update accumulated (only non-zero groups in this community)
        bc_comp = comp[best_community]
        nz_groups = np.nonzero(bc_comp)[0]
        for h in nz_groups:
            count_h = bc_comp[h]
            if h != g:
                accumulated[g, h] += count_h
                accumulated[h, g] += count_h
            else:
                accumulated[g, g] += 2 * count_h

        # Update community composition
        comp[best_community, g] += 1
        community_sizes[best_community] += 1

        # Assign node
        key = (best_community, g)
        if key not in G.communities_to_nodes:
            G.communities_to_nodes[key] = []
        G.communities_to_nodes[key].append(node)
        G.nodes_to_communities[node] = best_community
        if best_community not in G.communities_to_groups:
            G.communities_to_groups[best_community] = []
        G.communities_to_groups[best_community].append(g)

        if (node_idx + 1) % 500 == 0:
            print(f"Capacity assignment: {node_idx + 1}/{total_nodes} nodes "
                  f"({100*(node_idx+1)/total_nodes:.1f}%), {num_active} communities")

    G.number_of_communities = num_active
    # refine_node_assignments(G, target)

def refine_node_assignments(G, target, max_evals=5000):
    """
    Refines node assignments by moving nodes between communities to 
    minimize the global Mean Squared Error against the target.
    """

    n_groups = target.shape[0]
    # 1. Build initial community composition matrix
    max_comm = max(G.nodes_to_communities.values()) + 1
    comp = np.zeros((max_comm, n_groups))
    for node, comm in G.nodes_to_communities.items():
        g = G.nodes_to_group[node] # Assuming G stores the node's group
        comp[comm, g] += 1

    def get_global_error(composition):
        # Calculates global group-to-group counts
        # Error = sum( (Target - sum_over_communities(C_g * C_h)) ^ 2 )
        current_acc = np.zeros((n_groups, n_groups))
        for c in range(len(composition)):
            c_vec = composition[c].reshape(-1, 1)
            # Outer product represents all pairs in the community
            comm_edges = np.dot(c_vec, c_vec.T)
            # Diagonal adjustment: internal edges are counted differently in the original code
            np.fill_diagonal(comm_edges, composition[c] * (composition[c] - 1))
            current_acc += comm_edges
        return np.sum((target - current_acc) ** 2)

    current_error = get_global_error(comp)
    nodes = list(G.nodes_to_communities.keys())
    u_s = random.choices(nodes, k=max_evals)
    new_comms = random.choices([i for i in range(max_comm)], k=max_evals)
    for i in range(max_evals):
        # Pick a random node and a potential new community
        u = u_s[i]
        old_comm = G.nodes_to_communities[u]
        new_comm = new_comms[i]
        
        if old_comm == new_comm: continue
        
        g = G.nodes_to_group[u]
        
        # Simulate Move
        comp[old_comm, g] -= 1
        comp[new_comm, g] += 1
        
        new_error = get_global_error(comp)
        
        if new_error < current_error:
            # Keep the move
            current_error = new_error
            G.nodes_to_communities[u] = new_comm
        else:
            # Revert the move
            comp[old_comm, g] += 1
            comp[new_comm, g] -= 1

        if i % 100 == 0:
            print(f"Refinement Iteration {i}: Global Error = {current_error:.2f}")

    return G

def populate_communities_capacity(G, num_communities, community_size_distribution='natural'):
    """
    Assign nodes to communities by matching absolute edge counts against
    maximum_num_links budget, with feasibility constraints ensuring
    communities can be fully connected without exceeding the budget.

    Uses same SA temperature schedule as populate_communities() but with
    capacity-based distance (absolute edge counts, not probabilities).

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Initial number of communities (may grow if needed)
    community_size_distribution : str or array-like, optional
        Controls community size distribution
    """
    total_nodes = len(list(G.graph.nodes))
    n_groups = int(len(G.group_ids))

    # Build target matrix from maximum_num_links
    target = np.zeros((n_groups, n_groups), dtype=np.float64)
    for (i, j), count in G.maximum_num_links.items():
        target[i, j] = count

    # Compute probability matrix (needed for JSON serialization / load_communities)
    epsilon = 1e-5
    row_sums = target.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = epsilon
    G.probability_matrix = target / row_sums
    G.number_of_communities = num_communities

    # Target community sizes
    if isinstance(community_size_distribution, (list, np.ndarray)):
        target_sizes = np.array(community_size_distribution)
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'powerlaw':
        ranks = np.arange(1, num_communities + 1)
        sizes = 1.0 / (ranks ** 1)
        target_sizes = sizes / sizes.sum()
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'uniform':
        target_sizes = np.ones(num_communities) / num_communities
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    else:
        target_counts = None

    # Initialize community structures
    for community_idx in range(num_communities):
        for group_id in range(n_groups):
            G.communities_to_nodes[(community_idx, group_id)] = []
        G.communities_to_groups[community_idx] = []

    # Shuffle nodes
    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    # Try Rust backend, fall back to Python
    try:
        import X
        from asnu_rust import process_nodes_capacity
        print("Using Rust backend for capacity-based node processing...")

        budget = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}

        assignments = process_nodes_capacity(
            all_nodes.astype(np.int64),
            node_groups.astype(np.int64),
            budget,
            n_groups,
            num_communities,
            target_counts if target_counts is not None else None,
            total_nodes,
        )

        # Populate G structures from assignments
        for i in range(len(all_nodes)):
            node = int(all_nodes[i])
            comm = int(assignments[i])
            group = int(node_groups[i])
            key = (comm, group)
            if key not in G.communities_to_nodes:
                G.communities_to_nodes[key] = []
            G.communities_to_nodes[key].append(node)
            G.nodes_to_communities[node] = comm
            if comm not in G.communities_to_groups:
                G.communities_to_groups[comm] = []
            G.communities_to_groups[comm].append(group)

        G.number_of_communities = int(assignments.max()) + 1 if len(assignments) > 0 else 0
    except ImportError:
        print("Using Python fallback for capacity-based node processing...")

        _process_nodes_capacity_python(
            G, all_nodes, node_groups, num_communities,
            target, total_nodes, target_counts,
        )

    print(f"\nCapacity-based community assignment complete: "
          f"{total_nodes} nodes → {G.number_of_communities} communities")


def connect_all_within_communities(G, verbose=True):
    """
    Connect all nodes within each community to each other.

    Creates a fully connected graph within each community using vectorized
    operations for maximum efficiency.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community assignments
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Statistics about edges created
    """
    verbose = False
    if verbose:
        print("\nConnecting all nodes within communities...")

    stats = {
        'total_edges': 0,
        'edges_per_community': {}
    }

    # OPTIMIZED: Build community membership lookup once
    communities_nodes = [[] for _ in range(G.number_of_communities)]
    for node, comm in G.nodes_to_communities.items():
        communities_nodes[comm].append(node)

    # For each community, connect all nodes within it
    for community_id in range(G.number_of_communities):
        community_nodes = communities_nodes[community_id]

        if len(community_nodes) == 0:
            continue

        # Use itertools.product to generate all pairs efficiently
        # Filter out self-loops inline
        edges_to_add = [(src, dst) for src, dst in product(community_nodes, repeat=2)
                       if src != dst]

        # Batch add edges (much faster than individual add_edge calls)
        G.graph.add_edges_from(edges_to_add)

        edges_added = len(edges_to_add)
        stats['edges_per_community'][community_id] = edges_added
        stats['total_edges'] += edges_added

        # Progress reporting for large numbers of communities
        if (community_id + 1) % 5000 == 0 or community_id == 0:
            print(f"  Connected {community_id + 1}/{G.number_of_communities} communities ({(community_id + 1) / G.number_of_communities * 100:.1f}%)")

        if verbose:
            print(f"  Community {community_id}: {len(community_nodes)} nodes, {edges_added} edges")

    if verbose:
        print(f"  Total edges created: {stats['total_edges']}")

    return stats


def fill_unfulfilled_group_pairs(G, reciprocity_p, verbose=True):
    """
    Complete any group pairs that didn't reach their target edge count.

    Randomly creates edges between nodes from unfulfilled group pairs until
    targets are met or maximum attempts are reached.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with existing edges
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Statistics about the filling process
    """
    if verbose:
        print("\nFilling unfulfilled group pairs...")

    unfulfilled_pairs = []
    stats = {
        'total_pairs': 0,
        'fulfilled_pairs': 0,
        'unfulfilled_pairs': 0,
        'edges_added': 0,
        'reciprocal_edges_added': 0
    }

    # Identify which group pairs need more edges
    for (src_id, dst_id) in G.maximum_num_links.keys():
        existing = G.existing_num_links.get((src_id, dst_id), 0)
        maximum = G.maximum_num_links[(src_id, dst_id)]

        stats['total_pairs'] += 1

        if maximum == 0:
            continue

        # Only try to fill pairs that are genuinely under the target
        if existing < maximum:
            unfulfilled_pairs.append((src_id, dst_id, existing, maximum))
            stats['unfulfilled_pairs'] += 1
        else:
            stats['fulfilled_pairs'] += 1

    if verbose:
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Fulfilled: {stats['fulfilled_pairs']}")
        print(f"  Unfulfilled: {stats['unfulfilled_pairs']}")

    # Add random edges to complete unfulfilled pairs
    partially_filled = 0
    if unfulfilled_pairs:
        for src_id, dst_id, existing, maximum in unfulfilled_pairs:
            existing = G.existing_num_links.get((src_id, dst_id), 0)
            maximum = G.maximum_num_links.get((src_id, dst_id), 0)
            needed = maximum - existing
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])

            if not src_nodes or not dst_nodes:
                continue

            attempts = 0
            max_attempts = needed * 20
            edges_added_for_pair = 0

            while G.existing_num_links[(src_id, dst_id)] < maximum and attempts < max_attempts:
                src_node = random.choice(src_nodes)
                dst_node = random.choice(dst_nodes)

                # Add edge if valid (no self-loops, no duplicates)
                if src_node != dst_node and not G.graph.has_edge(src_node, dst_node):
                    G.graph.add_edge(src_node, dst_node)
                    edges_added_for_pair += 1
                    G.existing_num_links[(src_id, dst_id)] += 1
                    stats['edges_added'] += 1

                    # Reciprocity - same pattern as grn.py
                    if random.uniform(0,1) < reciprocity_p:
                        if( G.existing_num_links[(dst_id, src_id)] < G.maximum_num_links[(dst_id, src_id)] and
                           not G.graph.has_edge(dst_node, src_node)):
                            G.graph.add_edge(dst_node, src_node)
                            G.existing_num_links[(dst_id, src_id)] += 1
                            stats['reciprocal_edges_added'] += 1
                            if (dst_id == src_id):
                                edges_added_for_pair += 1
                                stats['edges_added'] += 1

                attempts += 1

    if verbose:
        print(f"  Edges added: {stats['edges_added']}")
        print(f"  Reciprocal edges added: {stats['reciprocal_edges_added']}")

    return stats


def create_communities(pops_path, links_path, scale, number_of_communities=None,
                       output_path='communities.json', community_size_distribution='natural',
                       pop_column='n', src_suffix='_src', dst_suffix='_dst',
                       link_column='n', min_group_size=0, verbose=True,
                       mode='probability'):
    """
    Create community assignments and save them to a JSON file.

    This is a standalone step that can be run independently from network
    generation. The output file can later be passed to generate() via
    the community_file parameter.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    scale : float
        Population scaling factor
    number_of_communities : int or None
        Number of communities to create. Required for mode='probability'.
        For mode='capacity', this is the initial count (may grow dynamically).
    output_path : str
        Path for the output JSON file
    community_size_distribution : str or array-like, optional
        Controls community size distribution (default 'natural')
    pop_column : str, optional
        Column name for population counts (default 'n')
    src_suffix : str, optional
        Suffix for source group columns (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns (default '_dst')
    link_column : str, optional
        Column name for link counts (default 'n')
    min_group_size : int, optional
        Minimum nodes per group after scaling (default 0)
    verbose : bool, optional
        Whether to print progress information
    mode : str, optional
        'probability' (default): match probability distributions
        'capacity': match absolute edge counts with feasibility constraints

    Returns
    -------
    str
        Path to the saved JSON file
    """
    import json
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    if verbose:
        print("="*60)
        print(f"COMMUNITY CREATION (mode={mode})")
        print("="*60)

    # Create a temporary graph and initialize nodes
    G = NetworkXGraph()
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes in {len(G.group_ids)} groups")

    # Compute maximum link counts (needed for affinity matrix)
    _compute_maximum_num_links(G, links_path, scale, src_suffix=src_suffix,
                                dst_suffix=dst_suffix, link_column=link_column,
                                verbose=verbose)

    # Create community structure
    if mode == 'capacity':
        if number_of_communities is None:
            number_of_communities = 100  # sensible default for capacity mode
        if verbose:
            print(f"\nCapacity-based assignment (initial {number_of_communities} communities)...")
        populate_communities_capacity(G, number_of_communities,
                                      community_size_distribution=community_size_distribution)
    else:
        if number_of_communities is None:
            raise ValueError("number_of_communities is required for mode='probability'")
        if verbose:
            print(f"\nAssigning nodes to {number_of_communities} communities...")
        populate_communities(G, number_of_communities,
                             community_size_distribution=community_size_distribution)

    # Serialize to JSON (convert numpy types to native Python types)
    data = {
        'number_of_communities': int(G.number_of_communities),
        'probability_matrix': G.probability_matrix.tolist(),
        'nodes_to_communities': {
            str(k): int(v) for k, v in G.nodes_to_communities.items()
        },
        'communities_to_nodes': {
            str(k): [int(n) for n in v] for k, v in G.communities_to_nodes.items()
        },
        'communities_to_groups': {
            str(k): [int(g) for g in v] for k, v in G.communities_to_groups.items()
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    if verbose:
        print(f"\nCommunity assignments saved to {output_path}")
        print("="*60 + "\n")

    return output_path


def create_hierarchical_community_file(
    household_community_file,
    pops_path,
    links_path,
    scale,
    target_num_communities,
    output_path,
    pop_column='n',
    src_suffix='_src',
    dst_suffix='_dst',
    link_column='n',
    verbose=True,
):
    """
    Create a community file where household communities are grouped into
    super-communities (e.g. 5000 household communities → 5 buren communities).

    Nodes in the same household community are guaranteed to be in the same
    super-community. Uses block assignment for locality preservation.

    Parameters
    ----------
    household_community_file : str
        Path to the household community JSON (from create_communities())
    pops_path : str
        Path to population CSV (for computing probability matrix)
    links_path : str
        Path to interaction CSV for the target layer
    scale : float
        Population scaling factor
    target_num_communities : int
        Number of super-communities to create
    output_path : str
        Path to write the output JSON
    """
    import json
    import math
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    # Load household community assignments
    with open(household_community_file, 'r', encoding='utf-8') as f:
        hh_data = json.load(f)

    hh_num_communities = hh_data['number_of_communities']
    hh_nodes_to_communities = {int(k): v for k, v in hh_data['nodes_to_communities'].items()}

    # Block assignment: group household communities into super-communities
    block_size = math.ceil(hh_num_communities / target_num_communities)

    super_nodes_to_communities = {}
    for node, hh_comm in hh_nodes_to_communities.items():
        super_comm = min(hh_comm // block_size, target_num_communities - 1)
        super_nodes_to_communities[node] = super_comm

    # Compute probability matrix from this layer's own link data
    G_temp = NetworkXGraph('_temp_hierarchical')
    init_nodes(G_temp, pops_path, scale, pop_column=pop_column)
    _compute_maximum_num_links(G_temp, links_path, scale,
                               src_suffix=src_suffix, dst_suffix=dst_suffix,
                               link_column=link_column, verbose=False)

    n_groups = len(G_temp.group_ids)
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G_temp.maximum_num_links.items():
        affinity[i, j] = count
    row_sums = affinity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probability_matrix = affinity / row_sums

    # Clean up temp directory
    import shutil, os
    if os.path.exists('_temp_hierarchical'):
        shutil.rmtree('_temp_hierarchical')

    # Write community JSON
    data = {
        'number_of_communities': target_num_communities,
        'probability_matrix': probability_matrix.tolist(),
        'nodes_to_communities': {str(k): int(v) for k, v in super_nodes_to_communities.items()},
        'communities_to_nodes': {},
        'communities_to_groups': {},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    if verbose:
        print(f"  Hierarchical communities: {hh_num_communities} household → "
              f"{target_num_communities} super-communities (block_size={block_size})")
        print(f"  Saved to {output_path}")

    return output_path


def load_communities(G, community_file_path):
    """
    Load community assignments from a JSON file into a NetworkXGraph object.

    Node-to-community assignments are loaded from the file, but
    communities_to_nodes and communities_to_groups are recalculated from
    the actual graph nodes. This ensures correctness when the graph has
    fewer groups than when the community file was created.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes already initialized
    community_file_path : str
        Path to the JSON file created by create_communities()
    """
    import json

    with open(community_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G.number_of_communities = data['number_of_communities']
    G.probability_matrix = np.array(data['probability_matrix'])

    # Load node-to-community assignments, keeping only nodes present in graph
    graph_nodes = set(G.graph.nodes)

    G.nodes_to_communities = {}
    for k, v in data['nodes_to_communities'].items():
        node = int(k)
        if node in graph_nodes:
            G.nodes_to_communities[node] = v

    unassigned = graph_nodes - set(G.nodes_to_communities.keys())
    if unassigned:
        print(f"Warning: {len(unassigned)} graph nodes have no community assignment")

    # Recalculate communities_to_nodes and communities_to_groups
    # from the actual graph nodes, so they reflect current groups
    communities_groups = {}

    for node, community_id in G.nodes_to_communities.items():
        group_id = G.nodes_to_group[node]
        key = (community_id, group_id)
        if key not in G.communities_to_nodes:
            G.communities_to_nodes[key] = []
        G.communities_to_nodes[key].append(node)

        if community_id not in communities_groups:
            communities_groups[community_id] = set()
        communities_groups[community_id].add(group_id)

    G.communities_to_groups = {
        comm: list(groups) for comm, groups in communities_groups.items()
    }
