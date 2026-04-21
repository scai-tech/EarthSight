from matplotlib.patches import Polygon
from typing import List, Union, Dict, Optional
import json
from datetime import datetime
from src.filter import Filter
import timeit
import random
from rtree import index
import numpy as np

'''
API Example:    
{
    <AOI> :   List[Polygons]                          // Area of interest (Narrow vs. Broad)
    <Priority Tier> : [1-10]                          // Defines latency threshold
    <Type> : recurring vs. one-time                   // Bookkeeping
    <Filter Categories> (opt): List[filter_id]        // How much compute required, may be none
    <Time> (opt): When
}
'''
class Query:
    def __init__(self, 
                 AOI: List[Polygon], 
                 priority_tier: int,
                 type: str,
                 filter_categories: Optional[List[List[str]]] = None, 
                 time: Optional[datetime] = None) -> None:
        """
        Initialize a Query object with parameters.
        
        Args:
            AOI: List of Polygon objects representing Areas of Interest
            priority_tier: Integer from 1-10 defining latency threshold
            type: String, either "recurring" or "one-time"
            filter_categories: Optional list of filter IDs, DNF formula
            time: Optional datetime object specifying when the query should execute
        """
        # Validate inputs
        if not all(isinstance(poly, Polygon) for poly in AOI):
            raise TypeError("AOI must be a list of Polygon objects")
        if not isinstance(priority_tier, int) or not (1 <= priority_tier <= 10):
            raise ValueError("Priority tier must be an integer between 1 and 10")
        if type not in ["recurring", "one-time"]:
            raise ValueError("Type must be either 'recurring' or 'one-time'")
        
        # Assign values
        self.AOI = AOI
        self.priority_tier = priority_tier
        self.type = type
        self.filter_categories = filter_categories or []
        self.time = time
        self._paths = [poly.get_path() for poly in self.AOI]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Query':
        """
        Create a Query object from a dictionary or JSON-like object.
        
        Args:
            data: Dictionary containing query parameters
            
        Returns:
            Query object initialized with the provided data
        """
        # Extract and validate required fields
        if 'AOI' not in data or 'Priority Tier' not in data or 'Type' not in data:
            raise ValueError("Missing required fields: AOI, Priority Tier, and Type are required")
        
        # Convert AOI data to Polygon objects
        # Assuming data['AOI'] contains coordinates for polygons
        polygons = []
        for poly_data in data['AOI']:
            # Assuming poly_data is a list of (x, y) coordinates
            polygons.append(Polygon(poly_data))
        
        # Parse optional time field
        time_obj = None
        if 'Time' in data and data['Time']:
            try:
                if isinstance(data['Time'], str):
                    time_obj = datetime.fromisoformat(data['Time'])
                elif isinstance(data['Time'], datetime):
                    time_obj = data['Time']
            except ValueError:
                raise ValueError("Invalid time format. Expected ISO format string or datetime object")
        
        return cls(
            AOI=polygons,
            priority_tier=data['Priority Tier'],
            type=data['Type'],
            filter_categories=data.get('Filter Categories', []),
            time=time_obj
        )
    
    
    def AOI_check(self, point):
        return any(path.contains_point(point) for path in self._paths)
    



class SpatialQueryEngine:
    def __init__(self):
        self.queries = []
        self.spatial_index = index.Index()
        self.idx_to_query = {}


    def load_queries(self, queries: List[Query]):
        """
        Load queries into the spatial index and store them in memory.
        
        Args:
            queries: List of Query objects to be indexed
        """
        for idx, query in enumerate(queries):
            self.add_query(query, idx)
        print(f"Loaded {len(queries)} queries into the spatial index.")

        
    def add_query(self, query, idx):
        self.queries.append(query)
        # Get the bounding box of all AOIs in the query
        for aoi in query.AOI:
            # Get the bounding box (minx, miny, maxx, maxy)
            bounds = aoi.get_extents().get_points()
            minx, miny = np.min(bounds, axis=0)
            maxx, maxy = np.max(bounds, axis=0)
            # Add to spatial index
            self.spatial_index.insert(idx, (minx, miny, maxx, maxy))
        self.idx_to_query[idx] = query
        
    def get_queries_at_coord(self, coord, min_pri=0, max_pri=10):
        # First find all potentially matching queries using the spatial index
        x, y = coord
        potential_matches = list(self.spatial_index.intersection((x, y, x, y)))
        
        # Then filter by exact point-in-polygon and priority
        results = set()
        for idx in potential_matches:
            query = self.idx_to_query[idx]
            if query.AOI_check(coord) and min_pri <= query.priority_tier <= max_pri:
                results.add(query)

        return results

def generate_example_queries(num_queries=100_000, num_filters=32):
    """Generate example queries with spatial indexing for faster lookups"""
    # Initialize the spatial query engine
    engine = SpatialQueryEngine()
    
    # Create filters
    for i in range(num_filters):
        Filter.add_filter(
            filter_id=i, 
            filter_name=f"Filter {i}", 
            filter_time=random.uniform(0.5, 2.0), 
            filter_pass_probs=random.uniform(0.1, 0.9)
        )

    # Load city data
    with open("apps/all_cities_2500000_10x10.json") as f:
        city_data = json.load(f)
    cities = list(city_data.keys())
    
    # Generate queries sequentially (avoids multiprocessing errors)
    for i in range(num_queries):
        # Randomly pick 3 cities
        selected_cities = random.sample(cities, 3)
        aois = [Polygon(city_data[city]['polygon']) for city in selected_cities]
        
        # Randomly pick filters
        num_random_filters = random.randint(0, 4)
        filters = [random.randint(0, num_filters-1) for _ in range(num_random_filters)]
        
        # Create query with random priority
        query = Query(aois, random.randint(1, 10), "recurring", filters)
        engine.add_query(query, i)
        
        # Optional progress indicator for large datasets
        if i % 10000 == 0 and i > 0:
            print(f"Generated {i} queries...")
    
    return engine

def run_benchmark(engine, coords, min_pri=0, max_pri=10):
    """Run benchmark tests on multiple coordinates"""
    results = []
    
    for i, coord in enumerate(coords):
        # Measure execution time
        time_taken = timeit.timeit(lambda: engine.get_queries_at_coord(coord, min_pri, max_pri), number=1)
        
        # Get actual results
        matching_queries = engine.get_queries_at_coord(coord, min_pri, max_pri)
        
        results.append({
            'coord': coord,
            'matches': len(matching_queries),
            'time': time_taken
        })
        
        print(f"Queries at Coord {coord}: {len(matching_queries)} (Time: {time_taken:.6f} seconds)")
    
    return results

if __name__ == "__main__":
    random.seed(0)
    
    print("Generating queries with spatial indexing...")
    start_time = timeit.default_timer()
    engine = generate_example_queries()
    gen_time = timeit.default_timer() - start_time
    print(f"Total Queries Generated: {len(engine.queries)} in {gen_time:.2f} seconds")
    
    # Test coordinates
    test_coords = [
        [41.8, -124.0],
        [-41.8, 124.0],
        [28.8, -144.0]
    ]
    
    print("\nRunning benchmark...")
    benchmark_results = run_benchmark(engine, test_coords)
    
    # Additional analysis
    if benchmark_results:
        avg_time = sum(r['time'] for r in benchmark_results) / len(benchmark_results)
        print(f"\nAverage query time: {avg_time:.6f} seconds")