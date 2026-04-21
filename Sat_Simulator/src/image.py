import random
from src.data import Data
from src.const import CONTEXT_SWITCH_OVERHEAD
from src.utils import Time
from . import log
from src.filter import Filter
import src.formula as fx
from matplotlib.patches import Polygon
import src.multitask_formula as mtl

SEED = 42 # for reproducibility

def evaluate_image_serval(formula, simulated_assignment, debug=False):
    """
    Evaluate an image using the Serval (baseline) strategy.

    Evaluates filters sequentially in formula order without intelligent ordering
    or early-stopping. Each conjunction is checked left-to-right; the first
    fully-satisfied term determines the image's priority.

    Args:
        formula: DNF formula as list of (term, priority) tuples.
        simulated_assignment: Dict mapping filter_id -> bool for simulated pass/fail.
        debug: If True, print per-filter evaluation details.

    Returns:
        tuple: (priority, compute_time) — priority of matched term (0 if none),
            and total filter execution time in seconds.
    """
    compute_time = 0
    filter_results = {}
    num_filters_evaluated = 0
    for filter_group, group_priority in formula:
        satisfied = True
        for filter_id, _ in filter_group:
            if filter_id not in filter_results:
                filter : Filter = Filter.get_filter(filter_id)
                compute_time += filter.time + CONTEXT_SWITCH_OVERHEAD
                num_filters_evaluated += 1
                filter_results[filter_id] = simulated_assignment[filter_id]

            if debug:
                print(f"Filter ID: {filter_id}, Result: {filter_results[filter_id]}")
            
            if not filter_results[filter_id]:
                satisfied = False
                break
            
        if satisfied:
            return group_priority, compute_time
    return 0, compute_time

    
def evaluate_image_earthsight(formula, simulated_assignment, unique_filters, registry, debug=False):
    """
    Evaluate an image using the EarthSight strategy with dynamic filter ordering.

    Uses either MTL-based multitask evaluation (if registry is provided) or
    STL-based greedy evaluation with confidence-based early stopping. Filters
    are ordered by elimination power to minimize computation.

    Args:
        formula: DNF formula as list of (term, priority) tuples.
        simulated_assignment: Dict mapping filter_id -> bool for simulated pass/fail.
        unique_filters: Set of filter IDs appearing in the formula.
        registry: ModelRegistry for MTL mode, or None for STL mode.
        debug: If True, print evaluation details.

    Returns:
        tuple: (priority, compute_time) — priority score and total compute time.
    """
    if len(formula) == 0:
        return 0, 0
                
    if registry:
        _assignment, compute_time, confidence, priority = mtl.evaluate_formula_dnf_multitask(formula, registry.copy(), upper_threshold=0.7, simulated_assignment=simulated_assignment, debug=False)
    else:
        _assignment, compute_time, confidence, priority = fx.evaluate_formula_dnf(formula, unique_filters, lower_threshold=0, upper_threshold=0.7, simulated_assignment=simulated_assignment, mode=2, verbose=debug)
                
    if debug:
        print(f"ES Assignment: {_assignment}")
        print(f"ES Confidence: {confidence}")        
    return priority, compute_time


def evaluate_image(formula, mode, registry = None, include_fnr = True, compare = False):
    """
    Top-level image evaluation dispatcher. Simulates filter pass/fail outcomes
    using deterministic seeding, optionally applies false negative rates, then
    delegates to the mode-appropriate evaluator (serval, earthsight, or fifo).

    Args:
        formula: DNF formula as list of (term, priority) tuples.
        mode: Evaluation strategy — "serval", "earthsight", or "fifo".
        registry: ModelRegistry for MTL mode (required if mode="earthsight" with MTL).
        include_fnr: If True, apply false negative rates to true-positive filter results.
        compare: If True, run both serval and earthsight and return the result for `mode`.

    Returns:
        tuple: (priority, compute_time, ground_truth_priority) where ground_truth_priority
            is the priority that would be assigned with perfect filter accuracy.
    """
    if len(formula) != 0 and random.random() < 0:
            return 0, 0, 0
        
    global SEED
    random.seed(SEED)
    SEED += 1

    simulated_assignment = {}
    unique_filters = set(filter for term in formula for filter, polarity in term[0])
    
    for filter in unique_filters:
        simulated_assignment[filter] = (random.random() < Filter.get_filter(filter).pass_probs['pass'])
    ground_truth_pri = fx.ground_truth_priority(formula, simulated_assignment)

    if mode == "fifo":
        if len(formula) == 0:
            return 1, 0, ground_truth_pri
        return 5, 0, ground_truth_pri

    if include_fnr:
        for filter in unique_filters:
            if simulated_assignment[filter]:
                fnr = Filter.get_filter(filter).false_negative_rate
                simulated_assignment[filter] = random.random() >= fnr # keep it true with 1 - false negative rate

    if not compare:
        if mode == "serval":
            priority, compute_time = evaluate_image_serval(formula, simulated_assignment)

        elif mode == "earthsight":
            priority, compute_time = evaluate_image_earthsight(formula, simulated_assignment, unique_filters, registry)
            
            if ground_truth_pri > 1 and ground_truth_pri > priority:
                log.Log("PRIORITIZATION ERROR", {"ground_truth_pri": ground_truth_pri, "computed_priority": priority})

        return priority, compute_time, ground_truth_pri
    
    if compare:
        serval_priority, serval_compute_time = evaluate_image_serval(formula, simulated_assignment)
        earthsight_priority, earthsight_compute_time = evaluate_image_earthsight(formula, simulated_assignment, unique_filters, registry)


        if mode == "serval":
            priority, compute_time = serval_priority, serval_compute_time
        elif mode == "earthsight":
            priority, compute_time = earthsight_priority, earthsight_compute_time

        return priority, compute_time, ground_truth_pri
            
class Image(Data):
    """
    Represents a satellite-captured image with associated metadata, priority scoring,
    and ground truth labels. Extends Data for packet-based transmission.

    Comparison operators are reversed so that PriorityQueue (min-heap) yields
    the highest-scored image first.

    Attributes:
        satellite: The satellite that captured this image (or None).
        time (Time): Capture timestamp.
        coord (list): [lat, lon] coordinates of the image center.
        size (int): Image size in bits.
        score (int): Computed priority score from filter evaluation (-1 = unscored).
        compute_time (float): Seconds spent evaluating filters on this image.
        descriptor (int): Ground truth priority label (-1 = unannotated, 0 = low, >1 = high).
        earliest_possible_transmit_time (Time): Earliest time the image could be downlinked.
        name (str): Human-readable image name.

    Class Attributes:
        id (int): Auto-incrementing image ID counter.
    """
    id = 0
    def __init__(self, size: int, time: 'Time', coord=[0,0],
                 name="", satellite = None):
        """
        Args:
            size: Image size in bits.
            time: Capture timestamp (Time object).
            coord: [lat, lon] coordinates of the image center.
            name: Human-readable name for this image.
            satellite: The satellite that captured this image.
        """
        super().__init__(size)
        self.satellite = satellite
        self.time = time
        self.coord = coord
        self.size = size
        self.id = Image.id
        self.score = -1 # priority score
        self.compute_time = 0 # time taken to compute the score
        Image.id += 1
        self.name = name
        self.descriptor = -1 # ground truth value
        self.earliest_possible_transmit_time = None

    def set_score(self, value):
        """Set the priority score for this image."""
        self.score = value

    @classmethod
    def set_id(cls, value):
        """Reset the class-level image ID counter to a specific value."""
        cls.id = value

    @staticmethod
    def from_dict(data):
        """
        Construct an Image from a dictionary containing 'region' as [min_x, min_y, max_x, max_y].

        Args:
            data: Dictionary with image parameters including a 'region' bounding box.

        Returns:
            Image: A new Image instance.
        """
        min_x, min_y, max_x, max_y = data['region']
        region = Polygon([(min_x, min_y), (max_x, min_y),
                         (max_x, max_y), (min_x, max_y)])
        return Image(
            **data,
            region=region
        )

    # To implement custom comparator (on the score) for the priority queue in the detector
    def __lt__(self, obj):
        """self < obj."""
        # Priority queue is a min heap while we want to put the highest score first
        # So we reverse the comparison
        return self.score > obj.score if not self.score == obj.score else self.time < obj.time

    def __le__(self, obj):
        """self <= obj."""
        return self < obj or self == obj

    def __eq__(self, obj):
        """self == obj."""
        return self.score == obj.score and self.time == obj.time

    def __ne__(self, obj):
        """self != obj."""
        return not self == obj

    def __gt__(self, obj):
        """self > obj."""
        return not self <= obj

    def __ge__(self, obj):
        """self >= obj."""
        return not self < obj

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return "{{imageId: {}, imageSize: {}, imageScore: {}, imageName: {}}}".format(self.id, self.size, self.score, self.name)
