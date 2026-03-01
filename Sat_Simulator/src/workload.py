"""Loader for workload.json — converts JSON data into Filter, Query, and Polygon objects.

Reads the workload definition file (referenceData/workload.json) and provides
functions to construct the runtime objects (Filter, Query, Polygon) used by the
simulator. All region, filter, query, and scenario data is defined in the JSON
file; this module handles only parsing and object construction.
"""

import json
import os
import math
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from matplotlib.patches import Polygon
from src.filter import Filter
from src.query import Query

_WORKLOAD_PATH = os.path.join(os.path.dirname(__file__), "..", "referenceData", "workload.json")
_cache = None

def load(path: Optional[str] = None) -> Dict:
    """Load and cache the workload JSON. Returns the parsed dict."""
    global _cache
    if _cache is None:
        path = path or _WORKLOAD_PATH
        if path is not None:
            with open(path, "r") as f:
                _cache = json.load(f)
        else:
            raise FileNotFoundError("Workload JSON path is not specified.")
    return _cache

def _build_polygons(coord_lists: List[List[List[float]]]) -> List[Polygon]:
    """Convert a list of coordinate arrays into Polygon objects."""
    return [Polygon(coords) for coords in coord_lists]

def _build_urban_polygons(data: Dict) -> List[Polygon]:
    """Build urban region polygons from city definitions and corridors."""
    regions = data["regions"]
    cities = regions["urban_cities"]
    core_r = regions["urban_core_radius"]
    metro_r = regions["urban_metro_radius"]
    polygons = []

    for city in cities:
        lat, lon = city["lat"], city["lon"]
        # Core
        polygons.append(Polygon([
            (lon - core_r, lat - core_r), (lon - core_r, lat + core_r),
            (lon + core_r, lat + core_r), (lon + core_r, lat - core_r)
        ]))
        # Metro
        polygons.append(Polygon([
            (lon - metro_r, lat - metro_r), (lon - metro_r, lat + metro_r),
            (lon + metro_r, lat + metro_r), (lon + metro_r, lat - metro_r)
        ]))

    for corridor_coords in regions["urban_corridors"]:
        polygons.append(Polygon(corridor_coords))

    return polygons

def get_region_polygons(region_name: str, data: Optional[Dict] = None) -> List[Polygon]:
    """Return the full list of Polygon objects for a named region type."""
    if data is None:
        data = load()
    if region_name == "urban":
        return _build_urban_polygons(data)
    return _build_polygons(data["regions"][region_name])

def resolve_regions(region_refs: List[Dict], data: Optional[Dict] = None) -> List[Polygon]:
    """Resolve a list of region references (with optional slicing) into Polygons.

    Each ref is either {"name": "river"} or {"name": "earthquake", "slice": [0, 3]}.
    """
    if data is None:
        data = load()
    polygons = []
    for ref in region_refs:
        full = get_region_polygons(ref["name"], data)
        if "slice" in ref:
            s = ref["slice"]
            start = s[0]
            end = s[1]
            full = full[start:end]
        polygons.extend(full)
    return polygons

def get_all_filters(hardware: str = "tpu") -> List[Filter]:
    """Return a list of Filter objects for the given hardware target."""
    data = load()
    hw_key = "tpu" if hardware == "tpu" else "gpu"
    filters = []
    for fid, fdef in data["filters"][hw_key].items():
        filters.append(Filter(
            fid,
            fdef["name"],
            fdef["cost"],
            {"pass": fdef["pass_prob"], "fail": fdef["fail_prob"]}
        ))
    return filters

def build_query(query_name: str, data: Optional[Dict] = None) -> Query:
    """Build a Query object from a named query definition in the workload."""
    if data is None:
        data = load()
    qdef = data["queries"][query_name]
    aoi = resolve_regions(qdef["regions"], data)

    time_obj = None
    if "time_offset_hours" in qdef:
        time_obj = datetime.now() + timedelta(hours=qdef["time_offset_hours"])
    elif "time_offset_days" in qdef:
        time_obj = datetime.now() + timedelta(days=qdef["time_offset_days"])

    return Query(
        AOI=aoi,
        priority_tier=qdef["priority"],
        type=qdef["type"],
        filter_categories=qdef["filter_dnf"],
        time=time_obj,
    )

def get_scenario_config(scenario_name: str) -> Dict:
    """Return the raw scenario config dict (queries list + workload_intensity)."""
    data = load()
    return data["scenarios"].get(scenario_name, {})

def run_scenario(scenario_name: str) -> List[Query]:
    """Build and return the list of Query objects for a named scenario."""
    data = load()
    config = data["scenarios"][scenario_name]
    queries = [build_query(qname, data) for qname in config["queries"]]
    print(f"{scenario_name.title()} Scenario Queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    return queries

def get_padding_query() -> Query:
    """Return the Query object used for schedule padding."""
    data = load()
    return build_query(data["padding_query"], data)

def get_padding_probability() -> float:
    """Return the padding probability from the workload config."""
    data = load()
    return data.get("padding_probability", 0.2)

def create_global_grid(coverage_percentage: float = 10.0) -> List[Polygon]:
    """Create a global grid of polygons covering the specified % of Earth's surface."""
    if not 0 <= coverage_percentage <= 100:
        raise ValueError("Coverage percentage must be between 0 and 100")

    cell_coverage_pct = 0.77
    num_cells_needed = math.ceil((coverage_percentage / 100) * (100 / cell_coverage_pct))
    grid_polygons = []

    strategic_locations = [
        (-74.5, 40.5, -73.5, 41.0), (-118.5, 33.5, -117.5, 34.5),
        (115.5, 39.5, 117.0, 40.5), (139.5, 35.5, 140.5, 36.5),
        (77.0, 28.5, 78.0, 29.0), (30.0, 30.0, 32.0, 32.0),
        (90.0, 23.0, 92.0, 25.0), (-122.5, 37.0, -121.5, 38.0),
        (106.5, 10.0, 107.5, 11.0), (-120.0, 35.0, -118.0, 37.0),
    ]

    strategic_limit = min(len(strategic_locations), max(1, int(num_cells_needed * 0.2)))
    for i in range(strategic_limit):
        lon_min, lat_min, lon_max, lat_max = strategic_locations[i]
        grid_polygons.append(Polygon([
            (lon_min, lat_min), (lon_min, lat_max),
            (lon_max, lat_max), (lon_max, lat_min)
        ]))

    remaining_cells = num_cells_needed - strategic_limit
    all_grid_cells = []
    for lon in range(-180, 180, 5):
        for lat in range(-90, 90, 5):
            if random.random() < 0.3:
                all_grid_cells.append((lon, lat))
    random.shuffle(all_grid_cells)

    for lon, lat in all_grid_cells[:remaining_cells]:
        grid_polygons.append(Polygon([
            (lon, lat), (lon, lat + 5), (lon + 5, lat + 5), (lon + 5, lat)
        ]))

    return grid_polygons

def run_coverage_scaling_scenario(coverage_percentages: Optional[List[float]] = None, focus_categories: Optional[List[str]] = None) -> List[Query]:
    """Run the coverage scaling scenario with multiple coverage percentages."""
    if coverage_percentages is None:
        coverage_percentages = [1.0, 5.0, 10.0, 25.0, 50.0]

    coverage_queries = []
    for percentage in coverage_percentages:
        priority = max(1, min(10, 10 - int(percentage / 10)))
        aoi = create_global_grid(percentage)

        if not focus_categories:
            focus_categories = ["general"]

        filter_dnf = []
        if "flood" in focus_categories or "general" in focus_categories:
            filter_dnf.append(["F1", "F2", "G1"])
        if "wildfire" in focus_categories or "general" in focus_categories:
            filter_dnf.append(["W1", "W2", "G2"])
        if "earthquake" in focus_categories or "general" in focus_categories:
            filter_dnf.append(["E1", "E4", "G4"])
        if "maritime" in focus_categories or "general" in focus_categories:
            filter_dnf.append(["S1", "S4", "G3"])
        if "airspace" in focus_categories or "general" in focus_categories:
            filter_dnf.append(["A1", "A4", "G1"])
        if "urban" in focus_categories or "general" in focus_categories:
            filter_dnf.append(["G4", "I1", "I4"])
        if not filter_dnf:
            filter_dnf = [["G1", "G2", "G3"], ["G4", "G5", "G6"], ["I4", "S1", "A1"]]

        query = Query(
            AOI=aoi,
            priority_tier=priority,
            type="recurring",
            filter_categories=filter_dnf,
            time=None,
        )
        coverage_queries.append(query)
        print(f"Created coverage query at {percentage}% of Earth's surface "
              f"(priority: {priority}, polygons: {len(query.AOI)})")

    return coverage_queries
