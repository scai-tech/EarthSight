class Filter:
    """
    Represents an ML model/computer vision filter that can be applied to satellite imagery.

    Each filter has an associated execution time, pass probability, and false negative rate.
    Filters are stored in a class-level registry keyed by filter_id and looked up during
    DNF formula evaluation to determine which ML models to run on captured images.

    Attributes:
        filter_id (str): Unique identifier for this filter (e.g., "F1", "W2", "S3").
        filter_name (str): Human-readable name (e.g., "Water Extent", "Ship Detection").
        time (float): Execution time in seconds for running this filter on one image.
        pass_probs (dict): Dictionary with key 'pass' mapping to the probability [0,1]
            that a random image passes this filter.
        false_negative_rate (float): Probability that a true-positive image is incorrectly
            classified as negative. Set per-mode in run.py (0.05 for MTL, 0.035 for STL).

    Class Attributes:
        filters (dict): Class-level registry mapping filter_id -> Filter instance.
    """
    filters = {}

    def __init__(self, filter_id, filter_name, filter_time, filter_pass_probs) -> None:
        """
        Args:
            filter_id: Unique identifier for this filter.
            filter_name: Human-readable name of the filter.
            filter_time: Execution time in seconds for this filter.
            filter_pass_probs: Dictionary with key 'pass' -> probability that an image passes.
        """
        self.filter_id = filter_id
        self.filter_name = filter_name
        self.time = filter_time
        self.pass_probs = filter_pass_probs
        self.false_negative_rate = 0.0

    @classmethod
    def add_filter(cls, filter_id, filter_name, filter_time, filter_pass_probs) -> None:
        """
        Create a new Filter and register it in the class-level registry.

        Args:
            filter_id: Unique identifier for this filter.
            filter_name: Human-readable name of the filter.
            filter_time: Execution time in seconds.
            filter_pass_probs: Dictionary with key 'pass' -> probability.
        """
        cls.filters[filter_id] = Filter(filter_id, filter_name, filter_time, filter_pass_probs)

    @classmethod
    def add_filters(cls, filters):
        """
        Register multiple pre-constructed Filter instances in the class-level registry.

        Args:
            filters: Iterable of Filter objects to register.
        """
        for filter in filters:
            cls.filters[filter.filter_id] = filter


    @classmethod
    def get_filter(cls, filter_id):
        """
        Look up a filter by its ID. If filter_id is a tuple, uses the first element.

        Args:
            filter_id: The filter ID string, or a tuple whose first element is the ID.

        Returns:
            Filter: The registered Filter instance.

        Raises:
            KeyError: If filter_id is not found in the registry.
        """
        if isinstance(filter_id, tuple):
            filter_id = filter_id[0]
        return cls.filters[filter_id]
    

    @classmethod
    def apply_to_all(cls, func):
        """
        Applies a function to all filters
        """
        for filter in cls.filters.values():
            func(filter)

def get_processing_coefficient(config: dict, altitude: float = 0) -> float:
    """
    Get the processing coefficient based on the hardware configuration.

    Args:
        config: Dictionary containing the simulation configuration, including 'hardware' key.

    Returns:
        float: The processing coefficient (e.g., 1.0 for GPU, 0.1 for TPU).
    """
    if config.get("hardware") == "tpu":
        return (29,12,0.3,22)
    else:
        return (29,12,0.3,25)