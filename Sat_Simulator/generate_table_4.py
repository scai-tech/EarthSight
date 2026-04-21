from typing import List
import pandas as pd #type: ignore
from src.filter import Filter
from src.workload import load, run_scenario, get_all_filters, get_scenario_config
from src.multitask_formula import create_model_registry_from_filters, evaluate_formula_dnf_multitask, ExactDNFEvaluator
from src.query import SpatialQueryEngine, Query

import numpy as np
import random
import src.formula as fx
from matplotlib.patches import Polygon
import src.multitask_formula as mtl

JETSON_SCALING_FACTOR = 1.33 / 12.08
TPU_SCALING_FACTOR = 4.15 / 12.08

def cheat(formula, simulated_assignment):
    """
    Calculates the minimum theoretical time to determine the formula's state.

    If the formula is False, it calculates the sum of the costs of the cheapest 
    filter that invalidates each term.
    
    If the formula is True, it finds the cost of the cheapest term that is true.
    """
    priority = fx.ground_truth_priority(formula, simulated_assignment)
    compute_time = 0

    if priority == 0:        
        for term, _ in formula:
            cheapest_invalidating_cost = float('inf')
            
            # Find all filters that invalidate this specific term
            for f, polarity in term:
                if simulated_assignment.get(f, False) != polarity:
                    # This filter's value doesn't match the assignment, so it makes the term false.
                    cost = Filter.get_filter(f).time
                    cheapest_invalidating_cost = min(cheapest_invalidating_cost, cost)
            
            # Add the cost of the single cheapest invalidating filter for this term to the total.
            # If a term is somehow valid (which shouldn't happen if priority is 0), its cost is infinity,
            # which would signal a logic error elsewhere.
            if cheapest_invalidating_cost != float('inf'):
                compute_time += cheapest_invalidating_cost

    else:
        # A DNF formula is true if AT LEAST ONE term is true.
        # To verify this, we just need to find the single cheapest term to evaluate that is true.
        min_true_term_cost = float('inf')
        for term, _ in formula:
            term_is_satisfied = True
            current_term_cost = 0
            
            for f, polarity in term:
                current_term_cost += Filter.get_filter(f).time
                if simulated_assignment.get(f, False) != polarity:
                    term_is_satisfied = False
                    break # This term isn't true, so stop evaluating its cost and move to the next term.
            
            if term_is_satisfied:
                # This term is true; see if it's the cheapest one we've found so far.
                min_true_term_cost = min(min_true_term_cost, current_term_cost)
                
        compute_time = min_true_term_cost

    return priority, compute_time

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
        added_noise=noise_sigma
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
        added_noise=noise_sigma
    )
    return priority, compute_time

def run_benchmark(queries: List[Query], registry) -> pd.DataFrame:
    random.seed(42)
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
        ("serval", simulate_serval),
        ("earthsight_stl", simulate_earthsight_stl),
        ("earthsight_multitask", lambda f, a: simulate_earthsight_multitask(f, a, registry))
        # ("exact", lambda f, a: simulate_exact(f, a, registry)),
        # ("oracle", cheat)
    ]:
        predicted, compute_time = sim_fn(formula, simulated_assignment)
        results['mode'].append(mode)
        results['compute_time'].append(compute_time)
        results['correct'].append(predicted == gp)
        results['false_positive_high_priority'].append(predicted > 2 and gp < 1)
        results['ground_truth_high_priority'].append(gp > 2)

    return pd.DataFrame(results)

if __name__ == "__main__":
    all_filters = get_all_filters()
    Filter.add_filters(all_filters)
    for f in all_filters:
        f.time = f.time

    # Create model registry
    registry, _ = create_model_registry_from_filters(all_filters)

    # Load scenario
    qe = SpatialQueryEngine()
    # qe.load_queries(q_natural + q_military)
    all_queries = run_scenario("naturaldisaster") + run_scenario("intelligence") + run_scenario("combined")
    qe.load_queries(all_queries)

    # Sample coordinates and fetch queries
    coordinates = [(random.uniform(-180, 180), random.uniform(-90, 90)) for _ in range(10000)]
    all_queries = []
    for coord in coordinates:
        queries = qe.get_queries_at_coord(coord, min_pri=1, max_pri=10)
        if queries:  # Only add if there are actual queries
            all_queries.append(queries)

    print(f"Total benchmarks to run: {len(all_queries)}")
    # Initialize an empty DataFrame to collect all results
    all_results = []

    # Run benchmark for each set of queries
    for i, queries in enumerate(all_queries):
        if i % 200 == 0:
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

        # Add speedup term relative to serval
        serval_mean_time = time_summary.loc['serval', 'mean']
        time_summary['speedup_vs_serval'] = serval_mean_time / time_summary['mean']

        # Apply scaling to all terms, then restore the relative speedup column
        time_summary_jetson = time_summary * JETSON_SCALING_FACTOR
        time_summary_jetson['speedup_vs_serval'] = time_summary['speedup_vs_serval'] 
        
        time_summary_tpu = time_summary * TPU_SCALING_FACTOR
        time_summary_tpu['speedup_vs_serval'] = time_summary['speedup_vs_serval']

        print("\n=== Compute Time Summary (Jetson) seconds per image ===")
        print(time_summary_jetson.round(2).to_string())

        print("\n=== Compute Time Summary (TPU) seconds per image ===")
        print(time_summary_tpu.round(2).to_string())

        # create results directory and file if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        

        with open("results/table4.txt", "w+") as f: 
            time_summary_jetson_str = time_summary_jetson.round(2).to_string()
            time_summary_tpu_str = time_summary_tpu.round(2).to_string()
            f.write("=== Compute Time Summary (Jetson) seconds per image ===\n")
            f.write(time_summary_jetson_str)
            f.write("\n\n=== Compute Time Summary (TPU) seconds per image ===\n")
            f.write(time_summary_tpu_str)

    else:
        print("No valid benchmarks were run.")