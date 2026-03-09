"""
Network generation module for PyPleNet NetworkX.

This module provides functions to generate large-scale population networks
using NetworkX in-memory graph storage. It creates nodes from population data and
establishes edges based on interaction patterns, with support for scaling,
reciprocity, and preferential attachment.

This NetworkX-based implementation is significantly faster than file-based
approaches for graphs that fit in memory.

Functions
---------
init_nodes : Initialize nodes in the graph from population data
init_links : Initialize edges in the graph from interaction data
generate : Main function to generate a complete network

Examples
--------
>>> graph = generate('population.csv', 'interactions.xlsx',
...                  fraction=0.4, scale=0.1, reciprocity_p=0.2)
>>> print(f"Generated network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
"""
import os
import shutil

from asnu.core.utils import (find_nodes, read_file, desc_groups, stratified_allocate)
from asnu.core.grn import establish_links
from asnu.core.graph import NetworkXGraph
from asnu.core.community import (
    build_group_pair_to_communities_lookup,
    connect_all_within_communities,
    fill_unfulfilled_group_pairs,
    load_communities
)


def _compute_maximum_num_links(G, links_path, scale, src_suffix='_src',
                                dst_suffix='_dst', link_column='n', verbose=True):
    """
    Compute maximum link counts for all group pairs using stratified allocation.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with group_ids and group metadata already initialized
    links_path : str
        Path to interactions file (CSV or Excel)
    scale : float
        Population scaling factor
    src_suffix : str, optional
        Suffix for source group columns (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns (default '_dst')
    link_column : str, optional
        Name of column containing link counts (default 'n')
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    pandas.DataFrame
        The raw link data DataFrame for downstream use
    """
    df_n_group_links = read_file(links_path)

    if verbose:
        print("Calculating link requirements...")

    group_ids = G.group_ids
    G.maximum_num_links = {(int(i), int(j)): 0 for i in group_ids for j in group_ids}

    # Build (key, original_value) pairs for stratified allocation
    items = []
    for _, row in df_n_group_links.iterrows():
        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}

        _, src_id = find_nodes(G, **src_attrs)
        _, dst_id = find_nodes(G, **dst_attrs)

        items.append(((src_id, dst_id), row[link_column]))

    allocations = stratified_allocate(items, scale)
    for key, count in allocations.items():
        G.maximum_num_links[key] = count

    if verbose:
        total_links = sum(G.maximum_num_links.values())
        target = int(scale * sum(v for _, v in items))
        print(f"Total requested links: {total_links} (target: {target})")

    return df_n_group_links


def init_nodes(G, pops_path, scale=1, pop_column='n'):
    """
    Initialize nodes from population data using stratified sampling.

    Uses stratified allocation to preserve demographic proportions while scaling.

    Parameters
    ----------
    G : NetworkXGraph
        Wrapper with G.graph (nx.DiGraph) and metadata
    pops_path : str
        Path to population file (CSV or Excel)
    scale : float, optional
        Population scaling factor (default 1)
    pop_column : str, optional
        Name of the column containing population counts (default 'n')
    """
    print(G.group_to_attrs.keys())
    group_desc_dict, characteristic_cols = desc_groups(pops_path, pop_column=pop_column)

    # Build (key, original_value) pairs for stratified allocation
    items = [(gid, info[pop_column]) for gid, info in group_desc_dict.items()]
    node_allocations = stratified_allocate(items, scale)
    # Create nodes
    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        G.group_to_attrs[group_id] = attrs
        n_nodes = node_allocations[group_id]
        G.group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))

        for _ in range(n_nodes):
            G.graph.add_node(node_id, **attrs)
            G.nodes_to_group[node_id] = group_id
            node_id += 1

    # Create attribute to group mapping
    for group_id, attrs in G.group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        G.attrs_to_group[attrs_key] = group_id

    # Initialize link tracking
    group_ids = list(G.group_to_attrs.keys())

    G.group_ids = group_ids
    G.existing_num_links = {(src, dst): 0 for src in group_ids for dst in group_ids}

def _setup_no_community_structure(G):
    """
    Create a single synthetic community containing all nodes.

    This allows the edge creation code to work unchanged when no community
    structure is desired — all nodes are placed in community 0.
    """
    import numpy as np

    n_groups = len(G.group_ids)
    G.number_of_communities = 1

    # Build probability matrix (normalized affinity) even without communities
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G.maximum_num_links.items():
        affinity[i, j] = count
    epsilon = 1e-5
    G.probability_matrix = affinity / (affinity.sum(axis=1, keepdims=True) + epsilon)

    # Single community (0) contains all groups and all nodes
    G.communities_to_groups[0] = list(G.group_ids)
    for group_id in G.group_ids:
        G.communities_to_nodes[(0, group_id)] = list(G.group_to_nodes[group_id])
    for node in G.graph.nodes:
        G.nodes_to_communities[node] = 0


def _run_edge_creation_python(G, links_path, fraction, reciprocity_p, transitivity_p,
                              verbose, src_suffix, dst_suffix, pa_scope,
                              bridge_probability=0, pre_seed_edges=None):
    """Pure-Python fallback for edge creation."""
    warnings = []
    df_n_group_links = read_file(links_path)

    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    total_rows = len(df_n_group_links)
    for idx, row in df_n_group_links.iterrows():
        if verbose and ((idx + 1) % 500 == 0 or idx == 0 or idx == total_rows - 1):
            print(f"\rProcessing row {idx + 1} of {total_rows}", end="")

        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        num_requested_links = G.maximum_num_links[(src_id, dst_id)]

        if not src_nodes or not dst_nodes:
            continue

        valid_communities = group_pair_to_communities.get((src_id, dst_id), [])

        link_success = establish_links(G, src_id, dst_id,
                                       num_requested_links, fraction, reciprocity_p,
                                       transitivity_p, valid_communities, pa_scope,
                                       bridge_probability=bridge_probability,
                                       number_of_communities=G.number_of_communities)

        if not link_success:
            existing_links = G.existing_num_links[(src_id, dst_id)]
            warnings.append(f"Groups ({src_id})-({dst_id}): {existing_links} exceeds target {num_requested_links}")

    if verbose:
        print()
        if warnings:
            print(f"\nWarnings ({len(warnings)} group pairs):")
            for warning in warnings[:10]:
                print(f"  {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")


def _run_edge_creation(G, links_path, fraction, reciprocity_p, transitivity_p,
                       verbose, src_suffix, dst_suffix, pa_scope,
                       bridge_probability=0, pre_seed_edges=None):
    """
    Run the edge creation loop using the community structure already set on G.
    Tries Rust backend, falls back to Python.
    """
    try:
        from asnu_rust import run_edge_creation as rust_edge_creation
    except ImportError:
        _run_edge_creation_python(G, links_path, fraction, reciprocity_p,
                                  transitivity_p, verbose, src_suffix, dst_suffix, pa_scope,
                                  bridge_probability=bridge_probability,
                                  pre_seed_edges=pre_seed_edges)
        return

    if verbose:
        print("Using Rust backend for edge creation...")

    df_n_group_links = read_file(links_path)
    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    # Build group_pairs list: (src_id, dst_id, target_link_count)
    group_pairs = []
    for _, row in df_n_group_links.iterrows():
        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}
        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)
        if not src_nodes or not dst_nodes:
            continue
        target = G.maximum_num_links[(src_id, dst_id)]
        group_pairs.append((src_id, dst_id, target))

    # Convert G data to plain dicts for Rust
    ctn = {(int(k[0]), int(k[1])): [int(n) for n in v]
           for k, v in G.communities_to_nodes.items()}
    ntg = {int(k): int(v) for k, v in G.nodes_to_group.items()}
    mnl = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}
    vcm = {(int(k[0]), int(k[1])): [int(c) for c in v]
           for k, v in group_pair_to_communities.items()}

    # Convert pre_seed_edges for Rust (list of (int, int) tuples or None)
    rust_pre_edges = None
    if pre_seed_edges:
        rust_pre_edges = [(int(u), int(v)) for u, v in pre_seed_edges]

    new_edges, link_counts = rust_edge_creation(
        group_pairs, vcm, mnl, ctn, ntg,
        fraction, reciprocity_p, transitivity_p,
        pa_scope, G.number_of_communities,
        bridge_probability,
        rust_pre_edges,
    )

    # Apply edges to the NetworkX graph
    G.graph.add_edges_from(new_edges)
    for src, dst, count in link_counts:
        G.existing_num_links[(src, dst)] = count

    if verbose:
        print(f"\n  Created {len(new_edges)} edges")


def generate(pops_path, links_path, preferential_attachment, scale, reciprocity,
             transitivity, base_path="graph_data", verbose=True,
             pop_column='n', src_suffix='_src', dst_suffix='_dst', link_column='n',
             fill_unfulfilled=True, fully_connect_communities=False,
             pa_scope='local', community_file=None, bridge_probability=0,
             pre_seed_edges=None):
    """
    Generate a population-based network using NetworkX.

    Creates a network by first generating nodes from population data, then
    establishing edges based on interaction patterns. Supports preferential
    attachment, reciprocity, transitivity, and community structure.

    Community structure is provided via a pre-computed JSON file created by
    create_communities(). If no community file is given, edges are created
    without any community structure.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    preferential_attachment : float
        Preferential attachment strength (0-1)
    scale : float
        Population scaling factor
    reciprocity : float
        Probability of reciprocal edges (0-1)
    transitivity : float
        Probability of transitive edges (0-1)
    base_path : str, optional
        Directory for saving graph (default "graph_data")
    verbose : bool, optional
        Whether to print progress information
    pop_column : str, optional
        Column name for population counts in pops_path (default 'n')
    src_suffix : str, optional
        Suffix for source group columns in links_path (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns in links_path (default '_dst')
    link_column : str, optional
        Column name for link counts in links_path (default 'n')
    fill_unfulfilled : bool, optional
        Whether to fill unfulfilled group pairs after initial link creation (default True)
    fully_connect_communities : bool, optional
        Whether to fully connect all nodes within each community, bypassing normal
        link formation process (default False). Requires community_file.
    pa_scope : str, optional
        Scope of preferential attachment popularity (default 'local'):
        - 'local': popularity stays within the community (intra-community)
        - 'global': popularity spreads across all communities (inter-community)
    community_file : str, optional
        Path to a JSON file with pre-computed community assignments (default None).
        Created by create_communities(). If not provided, the network is generated
        without any community structure.
    bridge_probability : float, optional
        Probability of routing an edge to a neighboring community (±1, circular
        wrapping). Creates wide bridges between adjacent communities, modeling
        people participating in multiple foci. Default 0 (no bridging).

    Returns
    -------
    NetworkXGraph
        Generated network with graph data and metadata
    """
    if verbose:
        print("="*60)
        print("NETWORK GENERATION")
        print("="*60)
        print("\nStep 1: Creating nodes from population data...")

    # Prepare output directory
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    G = NetworkXGraph(base_path)

    # Create nodes with stratified allocation
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes")
        print("\nStep 2: Creating edges from interaction patterns...")

    # Compute maximum link targets from the links file
    _compute_maximum_num_links(G, links_path, scale, src_suffix=src_suffix,
                               dst_suffix=dst_suffix, link_column=link_column,
                               verbose=verbose)

    # Invert preferential attachment for internal representation
    preferential_attachment_fraction = 1 - preferential_attachment

    if community_file is not None:
        # --- Load pre-computed communities from file ---
        if verbose:
            print("\nStep 2a: Loading communities from file...")

        load_communities(G, community_file)

        if verbose:
            print(f"  Loaded {G.number_of_communities} communities from {community_file}")

        # Pre-seed edges into the graph (for multiplex hierarchical generation)
        if pre_seed_edges:
            G.graph.add_edges_from(pre_seed_edges)
            for u, v in pre_seed_edges:
                src_g = G.nodes_to_group[u]
                dst_g = G.nodes_to_group[v]
                G.existing_num_links[(src_g, dst_g)] += 1
            if verbose:
                print(f"  Pre-seeded {len(pre_seed_edges)} edges into graph")

        if fully_connect_communities:
            if verbose:
                print("\nStep 2b: Fully connecting nodes within communities...")
            connect_all_within_communities(G, verbose=verbose)
        else:
            if verbose:
                print("\nStep 2b: Creating edges using community structure...")
            _run_edge_creation(G, links_path, preferential_attachment_fraction,
                               reciprocity, transitivity, verbose,
                               src_suffix, dst_suffix, pa_scope,
                               bridge_probability=bridge_probability,
                               pre_seed_edges=pre_seed_edges)

            if fill_unfulfilled:
                if verbose:
                    print("\nStep 3: Filling remaining unfulfilled group pairs...")
                fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    else:
        # --- No community structure ---
        if verbose:
            print("\nNo community file given.")
            print("  Generating edges without community structure...")

        _setup_no_community_structure(G)

        _run_edge_creation(G, links_path, preferential_attachment_fraction,
                           reciprocity, transitivity, verbose,
                           src_suffix, dst_suffix, pa_scope,
                           bridge_probability=bridge_probability)

        if fill_unfulfilled:
            if verbose:
                print("\nStep 3: Filling remaining unfulfilled group pairs...")
            fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    # Save to disk
    G.finalize()

    if verbose:
        # Calculate link fulfillment statistics
        total_requested = sum(G.maximum_num_links.values())
        total_created = G.graph.number_of_edges()
        fulfillment_rate = (total_created / total_requested * 100) if total_requested > 0 else 0

        # Count overfulfilled pairs
        overfulfilled = sum(1 for (src, dst) in G.maximum_num_links.keys()
                           if G.existing_num_links.get((src, dst), 0) > G.maximum_num_links[(src, dst)])

        print(f"\n{'='*60}")
        print(f"NETWORK GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Nodes: {G.graph.number_of_nodes()}")
        print(f"Edges: {G.graph.number_of_edges()}")
        print(f"\nLink Fulfillment:")
        print(f"  Requested: {total_requested}")
        print(f"  Created: {total_created}")
        print(f"  Difference: {total_created - total_requested:+d}")
        print(f"  Rate: {fulfillment_rate:.1f}%")
        if overfulfilled > 0:
            print(f"  Overfulfilled pairs: {overfulfilled}")
        print(f"\nSaved to: {base_path}")
        print(f"{'='*60}\n")

    return G