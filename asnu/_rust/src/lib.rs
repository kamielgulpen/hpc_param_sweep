use std::collections::{HashMap, HashSet};

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Port of the "process nodes" loop from populate_communities().
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, community_composition, community_sizes, group_exposure, ideal, target_counts=None, total_nodes=0))]
fn process_nodes<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    mut community_composition: PyReadwriteArray2<'py, f64>,
    mut community_sizes: PyReadwriteArray1<'py, i32>,
    mut group_exposure: PyReadwriteArray2<'py, f64>,
    ideal: PyReadonlyArray2<'py, f64>,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let all_nodes = all_nodes.as_array();
    let node_groups = node_groups.as_array();
    let ideal = ideal.as_array();

    let num_communities = community_composition.as_array().shape()[0];
    let n_groups = community_composition.as_array().shape()[1];

    let tc: Option<Array1<i32>> = target_counts.map(|t| t.as_array().to_owned());

    let mut rng = thread_rng();
    let mut assignments: Vec<i64> = Vec::with_capacity(total_nodes);

    let mut distances = vec![0.0f64; num_communities];
    let mut hyp_row = vec![0.0f64; n_groups];

    for node_idx in 0..total_nodes {
        let group = node_groups[node_idx] as usize;

        let comp = community_composition.as_array();
        let ge = group_exposure.as_array();

        for c in 0..num_communities {
            let mut hyp_total: f64 = 0.0;
            for g in 0..n_groups {
                let val = ge[[group, g]] + comp[[c, g]];
                hyp_row[g] = val;
                hyp_total += val;
            }
            if hyp_total < 1e-10 {
                hyp_total = 1e-10;
            }
            let mut dist_sq: f64 = 0.0;
            for g in 0..n_groups {
                let diff = (hyp_row[g] / hyp_total) - ideal[[group, g]];
                dist_sq += diff * diff;
            }
            distances[c] = dist_sq.sqrt();
        }

        if let Some(ref tc) = tc {
            let sizes = community_sizes.as_array();
            for c in 0..num_communities {
                if sizes[c] >= tc[c] {
                    distances[c] = f64::INFINITY;
                }
            }
        }

        let temperature: f64 = 1.0 - (node_idx as f64 / total_nodes as f64);
        let best_community: usize;

        if temperature > 0.05 {
            let valid: Vec<usize> = (0..num_communities)
                .filter(|&c| distances[c].is_finite())
                .collect();

            if valid.len() > 1 {
                let max_neg_d = valid
                    .iter()
                    .map(|&c| -distances[c] / (temperature + 1e-10))
                    .fold(f64::NEG_INFINITY, f64::max);

                let weights: Vec<f64> = valid
                    .iter()
                    .map(|&c| ((-distances[c] / (temperature + 1e-10)) - max_neg_d).exp())
                    .collect();

                let dist = WeightedIndex::new(&weights).unwrap();
                best_community = valid[dist.sample(&mut rng)];
            } else if valid.len() == 1 {
                best_community = valid[0];
            } else {
                best_community = distances
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
            }
        } else {
            best_community = distances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
        }

        assignments.push(best_community as i64);

        {
            let comp = community_composition.as_array();
            let mut ge = group_exposure.as_array_mut();
            for g in 0..n_groups {
                ge[[group, g]] += comp[[best_community, g]];
            }
            for g in 0..n_groups {
                if comp[[best_community, g]] > 0.0 {
                    ge[[g, group]] += 1.0;
                }
            }
        }

        {
            let mut comp = community_composition.as_array_mut();
            comp[[best_community, group]] += 1.0;
        }
        {
            let mut sizes = community_sizes.as_array_mut();
            sizes[best_community] += 1;
        }

        if (node_idx + 1) % 5000 == 0 {
            let pct = 100.0 * (node_idx + 1) as f64 / total_nodes as f64;
            println!(
                "Assigned {}/{} nodes ({:.1}%)",
                node_idx + 1,
                total_nodes,
                pct
            );
        }
    }

    let result = Array1::from(assignments);
    Ok(PyArray1::from_owned_array_bound(py, result))
}


/// Port of _run_edge_creation + establish_links while loop.
///
/// Processes all group pairs in one Rust call, maintaining the edge set
/// and adjacency list internally. Returns all new edges and final link counts.
#[pyfunction]
#[pyo3(signature = (group_pairs, valid_communities_map, maximum_num_links, communities_to_nodes, nodes_to_group, fraction, reciprocity_p, transitivity_p, pa_scope, number_of_communities, bridge_probability=0.0, pre_existing_edges=None))]
fn run_edge_creation(
    // List of (src_id, dst_id, target_link_count) for each group pair
    group_pairs: Vec<(i64, i64, i64)>,
    // (src_id, dst_id) -> [community_ids] (may have duplicates for weighting)
    valid_communities_map: HashMap<(i64, i64), Vec<i64>>,
    // (src_id, dst_id) -> max link count
    maximum_num_links: HashMap<(i64, i64), i64>,
    // (community_id, group_id) -> [node_ids]
    communities_to_nodes: HashMap<(i64, i64), Vec<i64>>,
    // node_id -> group_id
    nodes_to_group: HashMap<i64, i64>,
    // Parameters
    fraction: f64,
    reciprocity_p: f64,
    transitivity_p: f64,
    pa_scope: String,
    number_of_communities: i64,
    bridge_probability: f64,
    // Optional pre-existing edges (for multiplex pre-seeding)
    pre_existing_edges: Option<Vec<(i64, i64)>>,
) -> PyResult<(Vec<(i64, i64)>, Vec<(i64, i64, i64)>)> {
    let mut rng = thread_rng();

    // Internal graph state
    let mut edges: HashSet<(i64, i64)> = HashSet::new();
    let mut adjacency: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut new_edges: Vec<(i64, i64)> = Vec::new();

    // Popularity pools: (community_id, group_id) -> [node_ids]
    let mut popularity_pool: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    // Link counters
    let mut existing_num_links: HashMap<(i64, i64), i64> = HashMap::new();
    for &(src, dst) in maximum_num_links.keys() {
        existing_num_links.insert((src, dst), 0);
    }

    // Initialize internal state from pre-existing edges (multiplex pre-seeding)
    if let Some(ref pre_edges) = pre_existing_edges {
        for &(s, d) in pre_edges {
            edges.insert((s, d));
            adjacency.entry(s).or_default().push(d);
            // Count toward link budget (do NOT add to new_edges — already in graph)
            let s_group = *nodes_to_group.get(&s).unwrap_or(&-1);
            let d_group = *nodes_to_group.get(&d).unwrap_or(&-1);
            if s_group >= 0 && d_group >= 0 {
                *existing_num_links.entry((s_group, d_group)).or_insert(0) += 1;
            }
        }
        if !pre_edges.is_empty() {
            println!("  Rust: initialized with {} pre-existing edges", pre_edges.len());
        }
    }

    // Src node list cache per (community_id, group_id)
    let mut src_node_cache: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    let total_pairs = group_pairs.len();

    for (pair_idx, (src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        let src_id = *src_id;
        let dst_id = *dst_id;
        let target_link_count = *target_link_count;

        if (pair_idx + 1) % 500 == 0 || pair_idx == 0 || pair_idx == total_pairs - 1 {
            println!("Processing pair {} of {}", pair_idx + 1, total_pairs);
        }

        let possible_communities = match valid_communities_map.get(&(src_id, dst_id)) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };

        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);

        if num_links >= target_link_count {
            continue;
        }

        let max_attempts = target_link_count * 10;
        let mut attempts: i64 = 0;

        // Batch community selection
        let batch_size: usize = 10000;
        let pc_len = possible_communities.len();
        let mut community_batch: Vec<i64> = (0..batch_size)
            .map(|_| possible_communities[rng.gen_range(0..pc_len)])
            .collect();
        let mut batch_idx: usize = 0;

        while num_links < target_link_count && attempts < max_attempts {
            let community_id = community_batch[batch_idx];
            batch_idx += 1;

            if batch_idx >= batch_size {
                community_batch = (0..batch_size)
                    .map(|_| possible_communities[rng.gen_range(0..pc_len)])
                    .collect();
                batch_idx = 0;
            }

            // Get src nodes for this community
            let src_cache_key = (community_id, src_id);
            if !src_node_cache.contains_key(&src_cache_key) {
                let nodes = communities_to_nodes
                    .get(&src_cache_key)
                    .cloned()
                    .unwrap_or_default();
                src_node_cache.insert(src_cache_key, nodes);
            }
            let src_nodes = src_node_cache.get(&src_cache_key).unwrap();
            if src_nodes.is_empty() {
                attempts += 1;
                continue;
            }

            // Decide: bridge edge (dst from neighboring community) or normal
            let dst_community = if bridge_probability > 0.0
                && number_of_communities > 1
                && rng.gen::<f64>() < bridge_probability
            {
                let direction: i64 = if rng.gen::<bool>() { 1 } else { -1 };
                ((community_id + direction).rem_euclid(number_of_communities)) as i64
            } else {
                community_id
            };

            // Initialize popularity pool for (dst_community, dst_group)
            let pool_key = (dst_community, dst_id);
            if !popularity_pool.contains_key(&pool_key) {
                let dst_nodes = communities_to_nodes
                    .get(&pool_key)
                    .cloned()
                    .unwrap_or_default();
                if !dst_nodes.is_empty() {
                    let sample_size = ((dst_nodes.len() as f64) * fraction).ceil() as usize;
                    let sample_size = sample_size.min(dst_nodes.len());
                    let mut sampled = dst_nodes;
                    sampled.shuffle(&mut rng);
                    sampled.truncate(sample_size);
                    popularity_pool.insert(pool_key, sampled);
                } else {
                    popularity_pool.insert(pool_key, vec![]);
                }
            }

            let pool = popularity_pool.get(&pool_key).unwrap();
            if pool.is_empty() {
                attempts += 1;
                continue;
            }

            // Pick random src and dst
            let s = src_nodes[rng.gen_range(0..src_nodes.len())];
            let d = pool[rng.gen_range(0..pool.len())];

            if s != d && !edges.contains(&(s, d)) {
                edges.insert((s, d));
                adjacency.entry(s).or_default().push(d);
                new_edges.push((s, d));
                num_links += 1;
                existing_num_links.insert((src_id, dst_id), num_links);

                // Reciprocity
                if rng.gen::<f64>() < reciprocity_p {
                    let rev_existing = *existing_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                    let rev_max = *maximum_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                    if rev_existing < rev_max && !edges.contains(&(d, s)) {
                        edges.insert((d, s));
                        adjacency.entry(d).or_default().push(s);
                        new_edges.push((d, s));
                        *existing_num_links.entry((dst_id, src_id)).or_insert(0) += 1;
                        if dst_id == src_id {
                            num_links += 1;
                            existing_num_links.insert((src_id, dst_id), num_links);
                        }
                    }
                }

                // Preferential attachment
                if rng.gen::<f64>() > fraction && fraction != 1.0 {
                    if pa_scope == "global" {
                        for comm_id in 0..number_of_communities {
                            if rng.gen::<f64>() < (1.0 / number_of_communities as f64) * fraction {
                                let global_key = (comm_id, dst_id);
                                if let Some(p) = popularity_pool.get_mut(&global_key) {
                                    p.push(d);
                                }
                            }
                        }
                    } else {
                        if rng.gen::<f64>() > fraction {
                            if let Some(p) = popularity_pool.get_mut(&pool_key) {
                                p.push(d);
                                // Also add a random node from the full dst community
                                if let Some(dst_community_nodes) = communities_to_nodes.get(&pool_key) {
                                    if !dst_community_nodes.is_empty() {
                                        let dst_random_community_node = dst_community_nodes[rng.gen_range(0..dst_community_nodes.len())];
                                        p.push(dst_random_community_node);
                                    }
                                }
                            }
                        }
                    }
                }

                // Transitivity
                if transitivity_p >= rng.gen::<f64>() {
                    let neighbors: Vec<i64> = adjacency
                        .get(&d)
                        .cloned()
                        .unwrap_or_default();
                    for n in neighbors {
                        if s == n {
                            continue;
                        }
                        let n_id = match nodes_to_group.get(&n) {
                            Some(&id) => id,
                            None => continue,
                        };
                        let pair = (src_id, n_id);
                        let max_l = match maximum_num_links.get(&pair) {
                            Some(&v) => v,
                            None => continue,
                        };
                        let existing = *existing_num_links.get(&pair).unwrap_or(&0);
                        if existing < max_l && !edges.contains(&(s, n)) {
                            edges.insert((s, n));
                            adjacency.entry(s).or_default().push(n);
                            new_edges.push((s, n));
                            *existing_num_links.entry(pair).or_insert(0) += 1;

                            if n_id == dst_id {
                                num_links += 1;
                                existing_num_links.insert((src_id, dst_id), num_links);
                            }

                            // Reciprocity for transitive edge
                            if rng.gen::<f64>() < reciprocity_p {
                                let rev_pair = (n_id, src_id);
                                let rev_existing =
                                    *existing_num_links.get(&rev_pair).unwrap_or(&0);
                                let rev_max =
                                    *maximum_num_links.get(&rev_pair).unwrap_or(&0);
                                if !edges.contains(&(n, s)) && rev_existing < rev_max {
                                    edges.insert((n, s));
                                    adjacency.entry(n).or_default().push(s);
                                    new_edges.push((n, s));
                                    *existing_num_links.entry(rev_pair).or_insert(0) += 1;
                                    if n_id == src_id && src_id == dst_id {
                                        num_links += 1;
                                        existing_num_links
                                            .insert((src_id, dst_id), num_links);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            attempts += 1;
        }
    }

    // Convert existing_num_links to flat triples
    let links_out: Vec<(i64, i64, i64)> = existing_num_links
        .into_iter()
        .map(|((s, d), c)| (s, d, c))
        .collect();

    Ok((new_edges, links_out))
}


/// Capacity-based community assignment.
///
/// Same SA temperature schedule as `process_nodes`, but evaluates communities
/// by checking whether fully connecting a new node stays within the
/// `maximum_num_links` budget (absolute edge counts, not probabilities).
///
/// Optimized: flat 2D arrays, only evaluates active (non-empty) communities.
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, budget, n_groups, initial_num_communities, target_counts=None, total_nodes=0))]
fn process_nodes_capacity<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    initial_num_communities: usize,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _all_nodes = all_nodes.as_array();
    let node_groups = node_groups.as_array();
    let tc: Option<Array1<i32>> = target_counts.map(|t| t.as_array().to_owned());

    let mut rng = thread_rng();
    let mut assignments: Vec<i64> = Vec::with_capacity(total_nodes);

    // Pre-allocate with some extra room for growth
    let max_communities = initial_num_communities + initial_num_communities / 4;

    // Flat 2D array for community composition: comp[c * n_groups + h] = count
    let mut comp: Vec<i64> = vec![0; max_communities * n_groups];
    let mut community_sizes: Vec<i64> = vec![0; max_communities];
    let mut num_active: usize = 0; // Only iterate up to this index
    let mut capacity: usize = max_communities;

    // Flat 2D array for accumulated edge counts: acc[g * n_groups + h]
    let mut acc: Vec<i64> = vec![0; n_groups * n_groups];

    // Target matrix as flat array for cache-friendly access
    let mut target: Vec<i64> = vec![0; n_groups * n_groups];
    for (&(sg, dg), &count) in budget.iter() {
        let sg_u = sg as usize;
        let dg_u = dg as usize;
        if sg_u < n_groups && dg_u < n_groups {
            target[sg_u * n_groups + dg_u] = count;
        }
    }

    // Pre-cache target rows/cols for the current group (updated per node)
    let mut target_row_g: Vec<i64> = vec![0; n_groups]; // target[g, :]
    let mut target_col_g: Vec<i64> = vec![0; n_groups]; // target[:, g]
    let mut acc_row_g: Vec<i64> = vec![0; n_groups];    // acc[g, :]
    let mut acc_col_g: Vec<i64> = vec![0; n_groups];    // acc[:, g]

    // Reusable buffers
    let mut distances: Vec<f64> = vec![0.0; max_communities];

    for node_idx in 0..total_nodes {
        let g = node_groups[node_idx] as usize;

        // Cache target and accumulated rows/cols for group g
        for h in 0..n_groups {
            target_row_g[h] = target[g * n_groups + h];
            target_col_g[h] = target[h * n_groups + g];
            acc_row_g[h] = acc[g * n_groups + h];
            acc_col_g[h] = acc[h * n_groups + g];
        }

        // Evaluate only active (non-empty) communities
        let mut best_community: Option<usize> = None;
        let eval_count = num_active;

        for c_idx in 0..eval_count {
            let c_offset = c_idx * n_groups;
            let mut feasible = true;
            let mut dist_sq: f64 = 0.0;

            for h in 0..n_groups {
                let count_h = comp[c_offset + h];
                if count_h == 0 {
                    // No members of this group in community — distance from target unchanged
                    let rem_gh = (target_row_g[h] - acc_row_g[h]) as f64;
                    dist_sq += rem_gh * rem_gh;
                    let rem_hg = (target_col_g[h] - acc_col_g[h]) as f64;
                    dist_sq += rem_hg * rem_hg;
                    continue;
                }

                if h != g {
                    // Outgoing g->h: cost = count_h
                    let hyp_gh = acc_row_g[h] + count_h;
                    if hyp_gh > target_row_g[h] {
                        feasible = false;
                        break;
                    }
                    let rem = (target_row_g[h] - hyp_gh) as f64;
                    dist_sq += rem * rem;

                    // Incoming h->g: cost = count_h
                    let hyp_hg = acc_col_g[h] + count_h;
                    if hyp_hg > target_col_g[h] {
                        feasible = false;
                        break;
                    }
                    let rem = (target_col_g[h] - hyp_hg) as f64;
                    dist_sq += rem * rem;
                } else {
                    // Self-group g->g: cost = 2 * count_g
                    let hyp_gg = acc_row_g[g] + 2 * count_h;
                    if hyp_gg > target_row_g[g] {
                        feasible = false;
                        break;
                    }
                    let rem = (target_row_g[g] - hyp_gg) as f64;
                    dist_sq += rem * rem;
                    // Note: for self-group, row and col are the same pair, already counted
                }
            }

            if !feasible {
                distances[c_idx] = f64::INFINITY;
                continue;
            }

            // Optional size limit
            if let Some(ref tc) = tc {
                if c_idx < tc.len() && community_sizes[c_idx] as i32 >= tc[c_idx] {
                    distances[c_idx] = f64::INFINITY;
                    continue;
                }
            }

            distances[c_idx] = dist_sq.sqrt();
        }

        // Temperature-based selection (same SA as process_nodes)
        let temperature: f64 = 1.0 - (node_idx as f64 / total_nodes as f64);

        if temperature > 0.05 && eval_count > 0 {
            let valid: Vec<usize> = (0..eval_count)
                .filter(|&c| distances[c].is_finite())
                .collect();

            if valid.len() > 1 {
                let max_neg_d = valid
                    .iter()
                    .map(|&c| -distances[c] / (temperature + 1e-10))
                    .fold(f64::NEG_INFINITY, f64::max);

                let weights: Vec<f64> = valid
                    .iter()
                    .map(|&c| ((-distances[c] / (temperature + 1e-10)) - max_neg_d).exp())
                    .collect();

                let dist = WeightedIndex::new(&weights).unwrap();
                best_community = Some(valid[dist.sample(&mut rng)]);
            } else if valid.len() == 1 {
                best_community = Some(valid[0]);
            }
        } else if eval_count > 0 {
            // Greedy: pick minimum distance
            let min_idx = (0..eval_count)
                .filter(|&c| distances[c].is_finite())
                .min_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap());
            if let Some(idx) = min_idx {
                best_community = Some(idx);
            }
        }

        // If no feasible community, create a new one at num_active
        let chosen = match best_community {
            Some(c) => c,
            None => {
                let new_idx = num_active;
                // Grow arrays if needed
                if new_idx >= capacity {
                    let new_cap = capacity + capacity / 2 + 1;
                    comp.resize(new_cap * n_groups, 0);
                    community_sizes.resize(new_cap, 0);
                    distances.resize(new_cap, 0.0);
                    capacity = new_cap;
                }
                num_active += 1;
                new_idx
            }
        };

        // If chosen == num_active (first node going into this slot), bump num_active
        if chosen >= num_active {
            num_active = chosen + 1;
        }

        // Update accumulated edge counts
        {
            let c_offset = chosen * n_groups;
            for h in 0..n_groups {
                let count_h = comp[c_offset + h];
                if count_h == 0 {
                    continue;
                }
                if h != g {
                    acc[g * n_groups + h] += count_h;
                    acc[h * n_groups + g] += count_h;
                } else {
                    acc[g * n_groups + g] += 2 * count_h;
                }
            }
        }

        // Add node to community
        comp[chosen * n_groups + g] += 1;
        community_sizes[chosen] += 1;
        assignments.push(chosen as i64);

        if (node_idx + 1) % 5000 == 0 {
            let pct = 100.0 * (node_idx + 1) as f64 / total_nodes as f64;
            println!(
                "Capacity assignment: {}/{} nodes ({:.1}%), {} active communities",
                node_idx + 1,
                total_nodes,
                pct,
                num_active
            );
        }
    }

    println!(
        "Capacity-based assignment complete: {} nodes -> {} active communities",
        total_nodes,
        num_active
    );

    let result = Array1::from(assignments);
    Ok(PyArray1::from_owned_array_bound(py, result))
}


/// Python module
#[pymodule]
fn asnu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(run_edge_creation, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity, m)?)?;
    Ok(())
}
