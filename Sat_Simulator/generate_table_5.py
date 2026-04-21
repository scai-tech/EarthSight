from typing import List, Set
from itertools import combinations
import pandas as pd #type: ignore
from src.filter import Filter
from src.workload import run_scenario, get_all_filters
from src.multitask_formula import create_model_registry_from_filters, evaluate_formula_dnf_multitask, ExactDNFEvaluator
from src.query import SpatialQueryEngine, Query

import random
import src.formula as fx

TPU_SCALING_FACTOR = 2.15 / 4.14

def _term_is_satisfied(term, assignment):
    return all(assignment.get(fid, False) == polarity for fid, polarity in term)


def _multitask_subset_cost(filter_ids: Set[str], registry):
    cost = 0.0
    backbones = set()

    for fid in filter_ids:
        model = registry.get_model_by_filter_id(fid)
        if model:
            cost += model.execution_time
            if model.backbone is not None:
                backbones.add(model.backbone.name)
        else:
            cost += Filter.get_filter(fid).time

    for backbone_name in backbones:
        backbone = registry.get_model(backbone_name)
        if backbone:
            cost += backbone.execution_time

    return cost


def _minimum_cover_cost(candidates, required_terms, mandatory_filters, coverage, registry):
    if not required_terms:
        return _multitask_subset_cost(mandatory_filters, registry)

    uncovered = required_terms.copy()
    for fid in mandatory_filters:
        uncovered -= coverage.get(fid, set())

    if not uncovered:
        return _multitask_subset_cost(mandatory_filters, registry)

    optional = [fid for fid in candidates if fid not in mandatory_filters]
    best_cost = float('inf')

    for size in range(len(optional) + 1):
        for subset in combinations(optional, size):
            chosen = set(subset) | mandatory_filters
            covered = set()
            for fid in chosen:
                covered |= coverage.get(fid, set())

            if uncovered.issubset(covered):
                best_cost = min(best_cost, _multitask_subset_cost(chosen, registry))

    return best_cost


def cheat(formula, simulated_assignment, registry):
    """
    Clairvoyant oracle lower bound under the multitask cost model.

    The oracle knows all filter outcomes and computes the minimum filter set
    needed to certify the ground-truth priority.
    """
    priority = fx.ground_truth_priority(formula, simulated_assignment)
    indexed_terms = list(enumerate(formula))

    if priority == 0:
        required_terms = {idx for idx, _ in indexed_terms}
        coverage = {}
        candidates = set()

        for idx, (term, _) in indexed_terms:
            for fid, polarity in term:
                if simulated_assignment.get(fid, False) != polarity:
                    coverage.setdefault(fid, set()).add(idx)
                    candidates.add(fid)

        compute_time = _minimum_cover_cost(
            candidates=candidates,
            required_terms=required_terms,
            mandatory_filters=set(),
            coverage=coverage,
            registry=registry,
        )
        return priority, compute_time

    higher_priority_terms = {
        idx for idx, (_, term_priority) in indexed_terms if term_priority > priority
    }

    coverage = {}
    candidates = set()
    for idx, (term, _) in indexed_terms:
        if idx not in higher_priority_terms:
            continue
        for fid, polarity in term:
            if simulated_assignment.get(fid, False) != polarity:
                coverage.setdefault(fid, set()).add(idx)
                candidates.add(fid)

    satisfied_target_terms = [
        term
        for term, term_priority in (entry for _, entry in indexed_terms)
        if term_priority == priority and _term_is_satisfied(term, simulated_assignment)
    ]

    best_cost = float('inf')
    for witness_term in satisfied_target_terms:
        mandatory = {fid for fid, _ in witness_term}
        cost = _minimum_cover_cost(
            candidates=candidates | mandatory,
            required_terms=higher_priority_terms,
            mandatory_filters=mandatory,
            coverage=coverage,
            registry=registry,
        )
        best_cost = min(best_cost, cost)

    return priority, best_cost

def simulate_exact(formula, simulated_assignment, registry):
    unique_filters = set(f for term in formula for f, _ in term[0])
    new_registry = registry.deepcopy()
    new_registry.prune_irrelevant_models(unique_filters)

    evaluator = ExactDNFEvaluator(formula, new_registry)
    priority, compute_time = evaluator.evaluate(simulated_assignment, debug=False)
    return priority, compute_time

def simulate_serval(formula, simulated_assignment):
    compute_time = 0
    filter_results = {}

    for filter_group, group_priority in formula:
        for filter_id, _ in filter_group:
            if filter_id not in filter_results:
                f = Filter.get_filter(filter_id)
                compute_time += f.time
                filter_results[filter_id] = simulated_assignment[filter_id]
            if not filter_results[filter_id]:
                break
        else:
            return group_priority, compute_time
    return 0, compute_time

def simulate_earthsight_stl(formula, simulated_assignment, noise_sigma=0):
    unique_filters = set(f for term in formula for f, _ in term[0])
    _, compute_time, _, priority = fx.evaluate_formula_dnf(
        formula, unique_filters,
        lower_threshold=0.0,
        upper_threshold=1.0,
        simulated_assignment=simulated_assignment,
        mode=0,
        added_noise=0.0
    )
    return priority, compute_time

def simulate_earthsight_multitask(formula, simulated_assignment, registry, noise_sigma=0):
    filter_ids = set(f for term in formula for f, _ in term[0])
    # save original pass probabilities
    new_registry = registry.deepcopy()
    new_registry.prune_irrelevant_models(filter_ids)
    _, compute_time, _, priority = evaluate_formula_dnf_multitask(
        formula,
        new_registry,
        lower_threshold=0.0,
        upper_threshold=1.0,
        simulated_assignment=simulated_assignment,
        debug=False,
        added_noise=0.0
    )
    return priority, compute_time

def run_benchmark(queries: List[Query], registry) -> pd.DataFrame:
    results = {
        'mode': [],
        'compute_time': [],
        'correct': [],
        'false_positive_high_priority': [],
        'ground_truth_high_priority': [],
    }

    formula = []

    for q in queries:
        for f_seq in q.filter_categories:
            if not f_seq:
                continue  # Skip empty formulas
            term = ([(f, True) for f in f_seq], q.priority_tier)
            formula.append(term)

    if not formula:
        # Return empty DataFrame with correct columns if no formulas
        return pd.DataFrame(results)

    unique_filters = set(var for term, pri in formula for var, polarity in term)
    simulated_assignment = {
        f: (random.random() < Filter.get_filter(f).pass_probs['pass'])
        for f in unique_filters
    }

    gp = fx.ground_truth_priority(formula, simulated_assignment)

    for mode, sim_fn in [
        ("earthsight_mt", lambda f, a: simulate_earthsight_multitask(f, a, registry)),
        ("exact_mt", lambda f, a: simulate_exact(f, a, registry)),
        ("oracle_mt", lambda f, a: cheat(f, a, registry))
    ]:
        predicted, compute_time = sim_fn(formula, simulated_assignment)
        results['mode'].append(mode)
        results['compute_time'].append(compute_time)
        results['correct'].append(predicted == gp)
        results['false_positive_high_priority'].append(predicted > 2 and gp < 1)
        results['ground_truth_high_priority'].append(gp > 2)

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting program...")
    random.seed(42)

    all_filters = get_all_filters()
    Filter.add_filters(all_filters)
    for f in all_filters:
        f.pass_probs['pass'] = min(0.7, f.pass_probs['pass'] * 4)

    # Create model registry
    registry, _ = create_model_registry_from_filters(all_filters)

    # Load scenario
    qe = SpatialQueryEngine() 
    all_queries = run_scenario("naturaldisaster")
    qe.load_queries(all_queries)

    # Sample coordinates and fetch queries
    coordinates = [(random.uniform(-180, 180), random.uniform(-90, 90)) for _ in range(10000)]
    all_queries = []
    for coord in coordinates:
        queries = qe.get_queries_at_coord(coord, min_pri=1, max_pri=10)
        if queries:  # Only add if there are actual queries
            all_queries.append(queries)


    # Filter out queries with more than 15 filters to keep benchmarks manageable
    # all queries is List[Set[Query]]
    # <= 5 filters per query, or <12 total filters in the set

    all_queries = [q for q in all_queries if all(len(qi.filter_categories) <= 5 for qi in q)]
    all_queries = [q for q in all_queries if len(set(f for qi in q for f_seq in qi.filter_categories for f in f_seq)) < 12]
    print(f"Total benchmarks to run: {len(all_queries)}")
    # Initialize an empty DataFrame to collect all results
    all_results = []

    # Run benchmark for each set of queries
    for i, queries in enumerate(all_queries):
        if i % 1 == 0:
            print(f"Progress: {i/len(all_queries) * 100:.2f}%")
            
        df_result = run_benchmark(queries, registry.copy())
        if not df_result.empty:
            # Add instance ID for tracking
            df_result['instance_id'] = i
            all_results.append(df_result)
    
    # Combine all results
    if all_results:
        df_all_results = pd.concat(all_results, ignore_index=True)
        
        # Compute time summary
        time_summary = df_all_results.groupby('mode')['compute_time'].agg(['mean', 'std', 'median', 'min', 'max'])
        time_summary_tpu = time_summary * TPU_SCALING_FACTOR
        print("\n=== Compute Time Summary (TPU) seconds per image ===")
        print(time_summary_tpu.round(2).to_string())

        # print("\n=== Compute Time Summary (Unscaled) seconds per image ===")
        # print(time_summary.round(2).to_string())

        # create results directory and file if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)

        with open("results/table5.txt", "w+") as f: 
            # time_summary_jetson_str = time_summary_jetson.round(2).to_string()
            time_summary_tpu_str = time_summary_tpu.round(2).to_string()
            # time_summary_default = time_summary.round(2).to_string()
            f.write("\n\n=== Compute Time Summary (TPU) seconds per image ===\n")
            f.write(time_summary_tpu_str)

    else:
        print("No valid benchmarks were run.")