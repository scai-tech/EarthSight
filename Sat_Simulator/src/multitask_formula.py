"""
Multitask DNF formula evaluation for satellite imagery filter pipelines.

This module provides a framework for evaluating Disjunctive Normal Form (DNF)
formulas in a multitask learning context, where multiple classification models
(filters) share backbone feature extractors. It supports:

- A model hierarchy (BackboneModel -> ClassifierModel) that captures shared
  computation: a backbone's cost is paid only once for all of its child
  classifiers.
- A ModelRegistry that tracks model dependencies, execution state, and the
  mapping between logical filter IDs and concrete model instances.
- A greedy heuristic evaluator (evaluate_formula_dnf_multitask) that picks
  the next filter to run based on a score of (1 - p) / t, aiming to falsify
  terms cheaply.
- An exact dynamic-programming evaluator (ExactDNFEvaluator) that computes
  the truly optimal adaptive evaluation policy minimizing expected cost.
- Helper functions for propagating partial assignments through a DNF formula,
  detecting satisfied/falsified terms, and converting Filter objects into
  the Model framework.

DNF formula representation used throughout:
    formula = [(term, priority), ...]
    term    = [(filter_id, polarity), ...]
    where polarity is True/False indicating whether the filter must pass/fail
    and priority is a numeric value indicating the importance of that term.
"""

import math
import random
from copy import deepcopy
from collections import defaultdict
from src.filter import Filter

class Model:
    """Base class for all models (filters) in the multitask evaluation framework.

    A Model represents a single computational unit (e.g., a neural network
    inference step) that takes some amount of time to execute. Concrete
    subclasses -- BackboneModel and ClassifierModel -- add specific
    semantics such as parent/child dependency relationships and pass
    probabilities.

    Attributes:
        name (str): Unique identifier for this model.
        execution_time (float): Wall-clock cost (in arbitrary time units) of
            running this model.
    """
    def __init__(self, name, execution_time):
        self.name = name
        self.execution_time = execution_time
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

class BackboneModel(Model):
    """Shared feature-extraction backbone that must execute before its children.

    In a multitask architecture several lightweight ClassifierModel heads share
    a single expensive backbone (e.g., a CNN feature extractor). The backbone's
    execution cost is incurred at most once: when the first child classifier
    that depends on it is evaluated. Subsequent children reuse the already-
    computed features and only pay their own (much smaller) head cost.

    Attributes:
        child_models (list[ClassifierModel]): Classifier heads that depend on
            this backbone. Populated via :meth:`add_child`.
    """
    def __init__(self, name, execution_time):
        super().__init__(name, execution_time)
        self.child_models = []
        
    def add_child(self, child_model):
        """Add a child model that depends on this backbone"""
        self.child_models.append(child_model)
        child_model.backbone = self

class ClassifierModel(Model):
    """Lightweight classification head that produces a boolean pass/fail output.

    A ClassifierModel typically sits on top of a BackboneModel. Its own
    execution_time represents only the cost of the classification head; the
    backbone cost is tracked separately through the dependency relationship.
    The ModelRegistry uses ``pass_probability`` together with
    ``execution_time`` to compute greedy evaluation scores.

    Attributes:
        pass_probability (float): Prior probability in [0, 1] that this
            classifier returns True (pass).
        backbone (BackboneModel or None): The backbone this classifier depends
            on, set automatically by :meth:`BackboneModel.add_child`.
    """
    def __init__(self, name, execution_time, pass_probability):
        super().__init__(name, execution_time)
        self.pass_probability = pass_probability
        self.backbone = None  # Will be set if this model depends on a backbone

class ModelRegistry:
    """Central registry that manages models, their dependencies, and execution state.

    The registry maintains four key data structures:

    * ``_models``  -- maps model name -> Model instance.
    * ``_dependency_graph`` -- maps model name -> list of model names it
      depends on (currently only used for backbone dependencies).
    * ``_execution_status`` -- maps model name -> bool indicating whether the
      model has already been executed in the current evaluation episode.
    * ``_filter_to_model`` -- maps logical filter ID (e.g. ``"F1"``) to the
      concrete model name (e.g. ``"F1_Water_Extent"``).

    Together these allow the evaluators to determine the effective cost of
    running a model (which may include a not-yet-executed backbone), to track
    which models still need to run, and to translate between the filter IDs
    used in DNF formulas and the model objects that implement them.
    """
    def __init__(self):
        self._models = {}
        self._dependency_graph = defaultdict(list)
        self._execution_status = {}  # Tracks which models have been executed
        self._filter_to_model = {}   # Maps filter IDs to model names

    def deepcopy(self):
        """Return a full deep copy of the registry, including all models and state.

        Unlike :meth:`copy`, this duplicates every internal object so that
        mutations to the returned registry cannot affect the original. This is
        used by :class:`ExactDNFEvaluator` when exploring hypothetical
        execution branches.

        Returns:
            ModelRegistry: An independent deep copy of this registry.
        """
        return deepcopy(self)

    def prune_irrelevant_models(self, relevant_filter_ids):
        """Remove models not referenced by the current formula.

        Given the set of filter IDs that actually appear in a DNF formula,
        this method discards every model (and its associated execution status
        and dependency edges) that is neither a classifier for one of those
        filters nor a backbone required by such a classifier. The dependency
        graph is then rebuilt from scratch to stay consistent.

        This is an in-place operation that mutates the registry.

        Args:
            relevant_filter_ids (set[str]): Filter IDs present in the formula
                being evaluated (e.g. ``{"F1", "F3", "W2"}``).
        """
        relevant_model_names = set()
        old_model_count = len(self._models)

        
        # Identify relevant models based on filter IDs
        for fid in relevant_filter_ids:
            model = self.get_model_by_filter_id(fid)
            if model:
                relevant_model_names.add(model.name)
                if model.backbone:
                    relevant_model_names.add(model.backbone.name)
        
        # Remove irrelevant models from the registry
        self._models = {name: model for name, model in self._models.items() if name in relevant_model_names}
        self._execution_status = {name: status for name, status in self._execution_status.items() if name in relevant_model_names}

        # Print number of old models, number of relevant models, and pruned models
        relevant_model_count = len(self._models)
        pruned_count = old_model_count - relevant_model_count
        # print(f"Pruned models: {relevant_model_count} relevant out of {old_model_count} total (pruned {pruned_count})")

        # Rebuild dependency graph
        new_dependency_graph = defaultdict(list)
        for name in relevant_model_names:
            model = self.get_model(name)
            if isinstance(model, ClassifierModel) and model.backbone:
                new_dependency_graph[name].append(model.backbone.name)
        self._dependency_graph = new_dependency_graph
        
    def register_model(self, model):
        """Register a model and record its dependency edges.

        The model is added to ``_models``, its execution status is initialised
        to ``False``, and -- if it is a ClassifierModel with an assigned
        backbone -- an edge is added to ``_dependency_graph`` so the backbone
        is listed as a prerequisite.

        Args:
            model (Model): The model instance to register.
        """
        self._models[model.name] = model
        self._execution_status[model.name] = False
        
        # If it's a classifier with a backbone, update dependency graph
        if isinstance(model, ClassifierModel) and model.backbone:
            self._dependency_graph[model.name].append(model.backbone.name)
            
    def get_model(self, name):
        """Look up a model by its unique name.

        Args:
            name (str): The model's registered name.

        Returns:
            Model or None: The model instance, or ``None`` if no model with
            that name has been registered.
        """
        return self._models.get(name)
    
    def get_all_classifier_models(self):
        """Return all registered ClassifierModel instances.

        Returns:
            dict[str, ClassifierModel]: A dictionary mapping model name to
            ClassifierModel for every registered model whose type is
            ClassifierModel (BackboneModels and plain Models are excluded).
        """
        return {name: model for name, model in self._models.items() 
                if isinstance(model, ClassifierModel)}
    
    def get_effective_execution_time(self, model_name):
        """Calculate the total wall-clock cost of running a model right now.

        For a plain Model or BackboneModel this is simply its own
        ``execution_time``. For a ClassifierModel whose backbone has not yet
        been executed, the backbone's ``execution_time`` is added because
        running this classifier requires running its backbone first.

        Args:
            model_name (str): Name of the model to query.

        Returns:
            float: The effective execution time. Returns 0 if the model is
            not found in the registry.
        """
        model = self.get_model(model_name)
        if not model:
            return 0
            
        # Base execution time is the model's own time
        time = model.execution_time
        
        # If it's a classifier with a backbone that hasn't been executed yet
        if (isinstance(model, ClassifierModel) and 
            model.backbone and 
            not self._execution_status[model.backbone.name]):
            time += model.backbone.execution_time
            
        return time
    
    def mark_executed(self, model_name):
        """Mark a model as executed and update dependency edges.

        Sets the model's execution status to ``True``. If the model is a
        BackboneModel, it is also removed from the dependency lists of all
        its child classifiers so that those children become eligible for
        execution (i.e., appear in :meth:`get_executable_models`).

        Args:
            model_name (str): Name of the model that has just been executed.
                Silently returns if the name is not present in the registry.
        """
        if model_name not in self._execution_status:
            return
            
        self._execution_status[model_name] = True
        
        # If it's a backbone, update the dependency graph for all its children
        model = self.get_model(model_name)
        if isinstance(model, BackboneModel):
            for child in model.child_models:
                child_name = child.name
                if model_name in self._dependency_graph.get(child_name, []):
                    self._dependency_graph[child_name].remove(model_name)
    
    def get_executable_models(self):
        """Return names of all models that are ready to execute.

        A model is considered executable if it has not yet been executed and
        all of its dependencies (entries in ``_dependency_graph``) have been
        satisfied (i.e., the dependency list is empty). For a ClassifierModel
        this means its backbone has already been marked as executed.

        Returns:
            list[str]: Names of models whose dependencies are fully satisfied
            and that have not yet been run.
        """
        executable = []
        for name, model in self._models.items():
            if not self._execution_status[name] and not self._dependency_graph.get(name, []):
                executable.append(name)
        return executable
    
    def register_filter_model_mapping(self, filter_id, model_name):
        """Record a mapping from a logical filter ID to its model name.

        This allows the registry to translate the filter IDs used in DNF
        formula terms (e.g. ``"F1"``) into the full model names stored in
        ``_models`` (e.g. ``"F1_Water_Extent"``).

        Args:
            filter_id (str): The short filter identifier.
            model_name (str): The corresponding registered model name.
        """
        self._filter_to_model[filter_id] = model_name
        
    def get_model_by_filter_id(self, filter_id):
        """Look up a model using a logical filter ID.

        Args:
            filter_id (str): The short filter identifier (e.g. ``"F1"``).

        Returns:
            Model or None: The model mapped to this filter ID, or ``None``
            if no mapping exists or the mapped model is not registered.
        """
        model_name = self._filter_to_model.get(filter_id)
        if model_name:
            return self.get_model(model_name)
        return None
    
    def get_filter_id_from_model_name(self, model_name):
        """Reverse-lookup: find the filter ID that maps to a given model name.

        Iterates over the ``_filter_to_model`` mapping and returns the first
        filter ID whose value matches ``model_name``.

        Args:
            model_name (str): The full registered model name.

        Returns:
            str or None: The corresponding filter ID, or ``None`` if no
            mapping points to this model name.
        """
        # Extract filter ID from model name (e.g., "F1_Water_Extent" -> "F1")
        for filter_id, mapped_name in self._filter_to_model.items():
            if mapped_name == model_name:
                return filter_id
        return None
    
    def copy(self):
        """Create a lightweight copy of the registry suitable for evaluation.

        The ``_models`` and ``_filter_to_model`` dictionaries are shared by
        reference (since models themselves are not mutated during evaluation),
        while ``_dependency_graph`` and ``_execution_status`` are deep-copied
        so that marking models as executed in the copy does not affect the
        original registry.

        Returns:
            ModelRegistry: A partially-shared copy of this registry.
        """
        new_registry = ModelRegistry()
        new_registry._models = self._models # can stay the same
        new_registry._dependency_graph = deepcopy(self._dependency_graph)
        new_registry._execution_status = deepcopy(self._execution_status)
        new_registry._filter_to_model = self._filter_to_model # can stay the same
        return new_registry

def find_highest_satisfied_priority(formula):
    """
    Find the highest priority of any satisfied term in the formula.
    A term is satisfied if it has no literals left (empty term).
    """
    highest_priority = 0
    for term, priority in formula:
        if not term:  # Empty term means it's satisfied
            highest_priority = max(highest_priority, priority)
    return highest_priority

def find_highest_possible_priority(formula):
    """
    Find the highest priority of any term that could still be satisfied.
    """
    return max([priority for _, priority in formula], default=0)

def evaluate_formula_dnf_multitask(formula, model_registry : ModelRegistry, lower_threshold=0.0, upper_threshold=1.0, simulated_assignment=None, debug=False, added_noise=0.0):
    """Greedily evaluate a prioritised DNF formula under the multitask model.

    At each step the function selects the unevaluated filter with the highest
    ``(1 - p) / t`` score (i.e., highest chance of falsifying a term per unit
    time), executes it (including its backbone if not already run), propagates
    the result through the formula, and repeats until the formula is resolved.

    Backbone costs are amortised: when a classifier's backbone has already been
    executed for a sibling classifier, only the head cost is charged.

    If ``added_noise > 0``, Gaussian noise is added to every filter's pass
    probability before scoring to simulate estimation error; the original
    probabilities are restored before the function returns.

    Args:
        formula (list[tuple[list[tuple[str,bool]], float]]): The DNF formula
            as a list of ``(term, priority)`` pairs. Each term is a list of
            ``(filter_id, polarity)`` literals.
        model_registry (ModelRegistry): Registry containing all models
            referenced by the formula. Will be mutated (execution status).
        lower_threshold (float): Unused -- reserved for future pruning.
        upper_threshold (float): Unused -- reserved for future pruning.
        simulated_assignment (dict[str, bool] or None): Pre-determined
            ground-truth pass/fail values for each filter. If ``None``,
            values are sampled randomly according to each filter's pass
            probability.
        debug (bool): If ``True``, print a trace of each evaluation step.
        added_noise (float): Standard deviation of Gaussian noise added to
            each filter's pass probability before scoring (0 = no noise).

    Returns:
        tuple: A 4-tuple of:
            - assignment (dict[str, bool]): The filter evaluations performed.
            - total_time (float): Cumulative execution time spent.
            - confidence (float): Always 1.0 (deterministic simulation).
            - priority (float): The highest priority among satisfied terms,
              or 0 if no term was satisfied.
    """
    total_time = 0.0
    assignment = {}
    
    # Initialize with the original formula
    current_formula = formula.copy()
    variables = set(f for term, _ in formula for f, _ in term)
    if not simulated_assignment:
        simulated_assignment = {}
        for var in variables:
            simulated_assignment[var] = (random.random() < Filter.get_filter(var).pass_probs['pass'])

    if added_noise > 0.0:
        # save passprobs to a dict
        original_pass_probs = {}
        for fid in variables:
            f_obj : Filter = Filter.get_filter(fid)
            original_pass_probs[fid] = f_obj.pass_probs['pass']
            noisy_prob = original_pass_probs[fid] + random.gauss(0, added_noise)
            noisy_prob = max(0.0, min(1.0, noisy_prob))
            f_obj.pass_probs['pass'] = noisy_prob
    
    while current_formula:
        # Identify all filters in the current formula - handle the correct structure
        unique_filters = set()
        for term_tuple in current_formula:
            term = term_tuple[0]  # Get the term (list of filter conditions)
            for filter_condition in term:
                fid = filter_condition[0]  # Extract filter ID
                unique_filters.add(fid)
        
        # If we've already evaluated all filters, check the result
        if all(fid in assignment for fid in unique_filters):
            break
            
        # Calculate scores for remaining filters
        scores = {}
        for fid in unique_filters:
            if fid in assignment:
                continue  # Skip already evaluated filters
                
            model : ClassifierModel = model_registry.get_model_by_filter_id(fid)
            if model:
                p = model.pass_probability
                t = model.execution_time if model.execution_time > 0 else 0.01
                scores[fid] = (1 - p) / t
            else:
                f_obj : Filter = Filter.get_filter(fid)
                p = f_obj.pass_probs["pass"]
                t = f_obj.time if f_obj.time > 0 else 0.01
                scores[fid] = (1 - p) / t
        
        if not scores:
            break  # No more filters to evaluate
            
        # Select the highest scoring filter
        best_filter = max(scores.items(), key=lambda x: x[1])[0]
        
        # Execute the selected filter
        model = model_registry.get_model_by_filter_id(best_filter)
        
        # Run the backbone first if needed
        if model and model.backbone and not model_registry._execution_status.get(model.backbone.name, False):
            backbone : Model = model.backbone
            total_time += backbone.execution_time
            model_registry.mark_executed(backbone.name)
        
        # Determine execution time
        exec_time = model.execution_time if model else Filter.get_filter(best_filter).time
        
        # Simulate evaluation
        result = simulated_assignment.get(best_filter)
        total_time += exec_time
        assignment[best_filter] = result
        
        if debug:
            print(f"Evaluated {best_filter} = {result}, time so far: {total_time}")
        
        # Propagate the formula based on this result
        current_formula = propagate_formula(current_formula, best_filter, result)
        
        # Check if we can determine the result early
        satisfied_term = find_satisfied_term(current_formula, assignment)
        
        if satisfied_term is not None:
            # We found a satisfied term, return its priority
            return assignment, total_time, 1.0, satisfied_term[1]
        
        if not current_formula:
            # No satisfiable terms remain
            return assignment, total_time, 1.0, 0
    
    if added_noise > 0.0:
        # restore original pass probabilities
        for fid in variables:
            f_obj : Filter = Filter.get_filter(fid)
            f_obj.pass_probs['pass'] = original_pass_probs[fid]
    
    # After evaluating all necessary filters, find the highest priority of any satisfied term
    result_priority = find_highest_satisfied_priority(formula, assignment)
    return assignment, total_time, 1.0, result_priority

def propagate_formula(formula, filter_id, result):
    """
    Propagates a DNF formula after evaluating a filter.
    Returns the updated formula with simplified terms.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    updated_formula = []
    
    for term_tuple in formula:
        term = term_tuple[0]  # Extract the term (list of filter conditions)
        priority = term_tuple[1]  # Extract the priority
        
        new_term = []
        term_falsified = False
        
        for filter_condition in term:
            fid = filter_condition[0]  # Extract filter ID
            polarity = filter_condition[1]  # Extract polarity (True/False)
            
            if fid == filter_id:
                # If this filter doesn't match the expected value, the term is falsified
                if polarity != result:
                    term_falsified = True
                    break
                # If it does match, we don't need to include it in the new term
                continue
            else:
                # Keep other filters in the term
                new_term.append((fid, polarity))
        
        if not term_falsified:
            # If term is empty, it's satisfied; otherwise add the simplified term
            if not new_term:
                # Empty term means it's satisfied - return just this term
                return [([], priority)]
            else:
                updated_formula.append((new_term, priority))
    
    return updated_formula

def find_satisfied_term(formula, assignment):
    """
    Find a term that is already satisfied by the current assignment.
    Returns the term if found, None otherwise.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    for term_tuple in formula:
        term = term_tuple[0]  # Extract the term (list of filter conditions)
        priority = term_tuple[1]  # Extract the priority
        
        satisfied = True
        
        for filter_condition in term:
            fid = filter_condition[0]  # Extract filter ID
            polarity = filter_condition[1]  # Extract polarity (True/False)
            
            if fid not in assignment or assignment[fid] != polarity:
                satisfied = False
                break
        
        if satisfied:
            return term_tuple
    
    return None

def find_highest_satisfied_priority(formula, assignment):
    """
    Find the highest priority of any satisfied term in the original formula.
    
    Formula structure: [(term, priority), ...] where term is [(filter_id, polarity), ...]
    """
    highest_priority = 0
    
    for term_tuple in formula:
        term = term_tuple[0]  # Extract the term (list of filter conditions)
        priority = term_tuple[1]  # Extract the priority
        
        satisfied = True
        
        for filter_condition in term:
            fid = filter_condition[0]  # Extract filter ID
            polarity = filter_condition[1]  # Extract polarity (True/False)
            
            if fid not in assignment or assignment[fid] != polarity:
                satisfied = False
                break
        
        if satisfied and priority > highest_priority:
            highest_priority = priority
    
    return highest_priority

def create_model_registry_from_filters(all_filters : list[Filter]):
    """Convert Filter objects into BackboneModel/ClassifierModel pairs and register them.

    Filters are grouped into domain categories based on their ID prefix
    (e.g. ``"F"`` for flood, ``"W"`` for wildfire). For each non-empty
    category a shared BackboneModel is created whose execution time is
    1.2x the median filter time in that category. Each individual filter
    then becomes a ClassifierModel with ``execution_time = filter.time * 0.2``
    and ``pass_probability`` taken from the filter's pass_probs. The
    classifier is attached as a child of its category's backbone.

    Args:
        all_filters (list[Filter]): All Filter instances to convert.

    Returns:
        tuple[ModelRegistry, dict[str, str]]: A 2-tuple of:
            - registry (ModelRegistry): Fully populated registry with
              backbones, classifiers, dependency edges, and filter-to-model
              mappings.
            - filter_to_model_map (dict[str, str]): Mapping from each
              filter's ``filter_id`` to the registered model name.
    """
    registry = ModelRegistry()
    filter_to_model_map = {}
    
    # Group filters by their domain/category
    filter_categories = {
        "flood": [f for f in all_filters if f.filter_id.startswith("F")],
        "wildfire": [f for f in all_filters if f.filter_id.startswith("W")],
        "earthquake": [f for f in all_filters if f.filter_id.startswith("E")],
        "ship": [f for f in all_filters if f.filter_id.startswith("S") and not f.filter_id.startswith("SC")],
        "ship_class": [f for f in all_filters if f.filter_id.startswith("SC")],
        "aircraft": [f for f in all_filters if f.filter_id.startswith("A")],
        "general": [f for f in all_filters if f.filter_id.startswith("G")],
        "infrastructure": [f for f in all_filters if f.filter_id.startswith("I")],
        "security": [f for f in all_filters if f.filter_id.startswith("B")]
    }
    
    # Create backbone models for each category
    backbones = {}
    for category, filters in filter_categories.items():
        if filters:  # Only create backbone if there are filters in this category
            backbone_name = f"{category}_backbone"
            # Use average cost from the filters as backbone execution time
            # median cost
            median_cost = sorted(f.time for f in filters)[len(filters) // 2]
            backbone = BackboneModel(backbone_name, median_cost * 1.2)  # Backbone slightly slower than classifiers
            registry.register_model(backbone)
            backbones[category] = backbone
    
    # Create classifier models for each filter and attach to appropriate backbone
    for f in all_filters:
        model_name = f"{f.filter_id}_{f.filter_name.replace(' ', '_')}"
        # Determine which category this filter belongs to
        category = None
        for cat_name, filters in filter_categories.items():
            if f in filters:
                category = cat_name
                break
        
        if category:
            # Create classifier model
            classifier = ClassifierModel(
                model_name, 
                f.time * 0.2,  # Use the filter's cost as execution time
                f.pass_probs["pass"]  # Use the pass probability
            )
            
            # Connect to backbone
            if category in backbones:
                backbones[category].add_child(classifier)
            
            # Register model
            registry.register_model(classifier)
            
            # Map filter_id to model_name
            filter_to_model_map[f.filter_id] = model_name

    
    registry._filter_to_model = filter_to_model_map
    # Create dependencies between related models
    
    return registry, filter_to_model_map


import math
import random
from copy import deepcopy
from collections import defaultdict
from src.filter import Filter
# Assuming the Model classes and ModelRegistry from your code are here

class ExactDNFEvaluator:
    """Compute the optimal adaptive evaluation policy for a prioritised DNF formula.

    Unlike the greedy heuristic in :func:`evaluate_formula_dnf_multitask`,
    this class uses dynamic programming (DP) with memoization to find the
    policy that minimises the **expected** total execution time. The state
    space is the set of all possible partial assignments to the formula's
    filter variables; at each state the DP considers every unevaluated
    filter, computes the immediate cost (including backbone amortisation)
    and the expected future cost under both outcomes (pass / fail), and
    selects the action with the lowest total expected cost.

    Because the state space is exponential in the number of filters
    (2^n partial assignments), this evaluator is practical only for
    formulas with a moderate number of distinct filters (roughly <= 20).

    Typical usage::

        evaluator = ExactDNFEvaluator(formula, registry)
        priority, total_time = evaluator.evaluate(simulated_assignment=ground_truth)

    Attributes:
        formula (list): The DNF formula being evaluated.
        model_registry (ModelRegistry): Registry of models (mutated during
            :meth:`evaluate` to track execution status).
        memo (dict): Memoization cache mapping frozen assignment states to
            ``(min_expected_cost, best_filter_id)`` tuples.
        all_filter_ids (set[str]): All unique filter IDs referenced in the
            formula.
    """
    def __init__(self, formula, model_registry: ModelRegistry):
        """Initialise the evaluator for a given formula and model registry.

        Args:
            formula (list[tuple[list[tuple[str,bool]], float]]): The DNF
                formula as ``[(term, priority), ...]``.
            model_registry (ModelRegistry): Registry containing all models
                referenced by the formula.
        """
        self.formula = formula
        self.model_registry = model_registry
        # Memoization cache: state -> (min_expected_cost, best_action)
        self.memo = {}
        # Get all unique filter IDs mentioned in the formula
        self.all_filter_ids = set(
            fid for term, _ in formula for fid, _ in term
        )
        # print(self.all_filter_ids)

    def _get_formula_status(self, assignment: dict):
        """
        Determines the formula's status given a partial assignment of filter values.

        Returns:
            - ("SATISFIED", priority): If any term is satisfied.
            - ("FALSIFIED", 0): If all terms are falsified.
            - ("UNDETERMINED", 0): If the outcome is not yet known.
        """
        highest_satisfied_priority = 0
        all_terms_falsified = True

        for term, priority in self.formula:
            term_satisfied = True
            term_falsified = False
            
            for fid, polarity in term:
                if fid in assignment:
                    if assignment[fid] != polarity:
                        # This literal is false, so the term is falsified
                        term_falsified = True
                        term_satisfied = False
                        break
                else:
                    # An unevaluated literal exists, so this term is not yet satisfied
                    term_satisfied = False

            if term_satisfied:
                # If we find any satisfied term, we can potentially stop early
                highest_satisfied_priority = max(highest_satisfied_priority, priority)

            # A term is only considered falsified if one of its literals is known to be false.
            # If it's just undetermined, we can't say the whole formula is falsified yet.
            if not term_falsified:
                all_terms_falsified = False

        if highest_satisfied_priority > 0:
            return "SATISFIED", highest_satisfied_priority
        
        if all_terms_falsified:
            return "FALSIFIED", 0

        return "UNDETERMINED", 0

    def _find_min_expected_cost(self, assignment_tuple: tuple):
        """
        The core recursive function for the dynamic programming approach.
        
        Args:
            assignment_tuple: A sorted tuple of (filter_id, value) pairs, representing the current state.
        
        Returns:
            (float, str): A tuple of (minimum_expected_future_cost, best_filter_to_evaluate).
        """
        if len(assignment_tuple) > len(self.all_filter_ids):
            print(assignment_tuple)
            print("SDFSDFSDFSDFSDFSDFSD")
            return (0, None)  # All filters evaluated, no future cost
        # Use a tuple for the state so it's hashable for the memoization cache
        if assignment_tuple in self.memo:
            return self.memo[assignment_tuple]

        assignment = dict(assignment_tuple)
        status, _ = self._get_formula_status(assignment)

        # Base case: If the formula's value is already determined, future cost is 0.
        if status in ["SATISFIED", "FALSIFIED"]:
            return (0, None)
            
        unevaluated_fids = self.all_filter_ids - assignment.keys()

        # Another base case: if there are no more filters to evaluate.
        if not unevaluated_fids:
            return (0, None)

        min_cost = float('inf')
        best_action = None
        # if len(self.memo) % 10000 == 0 and self.memo:
        #     print(f"Memo size: {len(self.memo)}")

        # Iterate over all possible next actions (evaluating an unevaluated filter)
        for fid in unevaluated_fids:
            model = self.model_registry.get_model_by_filter_id(fid)
            if fid not in self.all_filter_ids:
                print('error fid not in applicable', fid)
                continue

            
            # Create a temporary registry to calculate costs without modifying the main one
            temp_registry = self.model_registry.deepcopy()
            # Mark already executed models based on the current assignment
            for known_fid in assignment.keys():
                m = temp_registry.get_model_by_filter_id(known_fid)
                if m:
                    temp_registry.mark_executed(m.name)
                    if m.backbone:
                        temp_registry.mark_executed(m.backbone.name)

            immediate_cost = temp_registry.get_effective_execution_time(model.name)
            p_pass = model.pass_probability
            
            # --- Recursive step for the two possible outcomes ---
            # Outcome 1: Filter passes (returns True)
            assignment_if_pass = tuple(sorted(list(assignment.items()) + [(fid, True)]))
            cost_if_pass, _ = self._find_min_expected_cost(assignment_if_pass)

            # Outcome 2: Filter fails (returns False)
            assignment_if_fail = tuple(sorted(list(assignment.items()) + [(fid, False)]))
            cost_if_fail, _ = self._find_min_expected_cost(assignment_if_fail)
            
            # Calculate total expected cost for choosing this filter
            expected_cost = immediate_cost + p_pass * cost_if_pass + (1 - p_pass) * cost_if_fail
            
            if expected_cost < min_cost:
                min_cost = expected_cost
                best_action = fid
        
        # Memoize and return the result for the current state
        self.memo[assignment_tuple] = (min_cost, best_action)
        return min_cost, best_action

    def evaluate(self, simulated_assignment=None, debug=False):
        """Execute the optimal policy on a (simulated) instance.

        Starting from an empty assignment, this method repeatedly queries
        :meth:`_find_min_expected_cost` to determine the optimal next filter
        to evaluate, simulates its execution (charging the effective time
        including any backbone cost), records the outcome, and continues
        until the formula is either satisfied or fully falsified.

        The DP table is built lazily on the first call; subsequent calls with
        different ``simulated_assignment`` values reuse the cached policy.

        Args:
            simulated_assignment (dict[str, bool] or None): Ground-truth
                pass/fail values for each filter. If ``None``, outcomes are
                sampled randomly using each model's pass_probability.
            debug (bool): If ``True``, print a step-by-step execution trace.

        Returns:
            tuple[float, float]: A 2-tuple of:
                - priority (float): The highest priority among satisfied
                  terms, or 0 if no term was satisfied.
                - total_time (float): Cumulative execution time spent.
        """
        assignment = {}
        total_time = 0.0

        while True:
            status, priority = self._get_formula_status(assignment)
            if status != "UNDETERMINED":
                if debug: print(f"Formula decided. Status: {status}, Priority: {priority}")
                return priority, total_time

            assignment_tuple = tuple(sorted(assignment.items()))
            _, best_filter_to_run = self._find_min_expected_cost(assignment_tuple)

            if best_filter_to_run is None:
                # Should not happen if formula is still undetermined, but as a safeguard
                if debug: print("No more actions to take, but formula still undetermined.")
                break

            # --- Simulate executing the best filter ---
            model = self.model_registry.get_model_by_filter_id(best_filter_to_run)
            exec_time = self.model_registry.get_effective_execution_time(model.name)
            total_time += exec_time

            # Mark model and its backbone (if any) as executed
            if model.backbone:
                self.model_registry.mark_executed(model.backbone.name)
            self.model_registry.mark_executed(model.name)

            # Get simulated result
            p_pass = model.pass_probability
            result = simulated_assignment.get(best_filter_to_run) if simulated_assignment else (random.random() < p_pass)
            assignment[best_filter_to_run] = result
            
            if debug:
                print(f"Optimal action: Evaluate {best_filter_to_run}. Cost: {exec_time:.2f}. Outcome: {result}.")
                print(f"  Total time: {total_time:.2f}. Current assignment: {assignment}")
        
        # Final check after loop exits
        status, final_priority = self._get_formula_status(assignment)
        if debug:
            print("final", status, final_priority, total_time)
        return final_priority, total_time