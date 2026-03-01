"""
Entry point for the EarthSight satellite constellation simulator.

Configures and runs a full simulation comparing scheduling strategies (Serval
baseline vs. EarthSight ML-augmented) across different scenarios, learning
paradigms (MTL vs. STL), and hardware targets (Edge TPU vs. GPU).

Usage:
    cd Sat_Simulator
    python run.py --mode earthsight --scenario naturaldisaster --learning mtl --hardware tpu

CLI Arguments:
    --mode        Scheduling algorithm: "serval" (baseline) or "earthsight" (ML-augmented).
                  Default: "serval".
    --scenario    Query scenario to simulate:
                  "naturaldisaster" — flood/wildfire/earthquake monitoring regions.
                  "intelligence" — geopolitical region monitoring.
                  "combined" — hybrid of both.
                  "coverage_scaling" — constellation size scaling tests.
                  Default: "naturaldisaster".
    --learning    Learning paradigm: "mtl" (multitask, shared backbones) or
                  "stl" (single-task, independent filters). Default: "mtl".
    --hardware    Target on-board hardware: "tpu" (Edge TPU, 2W compute) or
                  "gpu" (30W compute). Default: "tpu".

Outputs:
    stdout — Summary statistics: percent annotated, delay by priority level
             (mean, std dev, percentiles), computation times, power metrics,
             and queue length statistics.
    log/rcv_data.json — Per-satellite reception delay data.
    log/compute_queues.json — Per-satellite computation queue size over time.

Simulation Flow:
    1. Load filters for the chosen hardware and register them globally.
    2. Set false negative rates (0.05 for MTL+EarthSight, 0.035 otherwise).
    3. Build MTL model registry if using multitask learning.
    4. Load scenario queries defining areas of interest and priorities.
    5. Load satellite TLEs and ground station locations from reference data.
    6. Wrap satellites with EarthsightSatellite decorators and set compute power.
    7. Initialize Simulator with 60-second timestep over a 9-hour window.
    8. Run the simulation loop.
    9. Compute and print analytics (delay distributions, computation times,
       power generation/consumption, queue lengths).
"""
from typing import List
import pandas as pd
from src.metrics import Metrics
from src.const import JETSON_POWER_DRAW, CORAL_POWER_DRAW
from src.station import Station
from src.satellite import Satellite
from src.earthsightgs import EarthSightGroundStation
from src.receiveGS import assess_gs_logs
from src.utils import Time, Location, TeeStream, get_mode_int
from src.simulator import Simulator
from src.earthsightsatellite import EarthsightSatellite
from src.scheduler import EarthSightScheduler
from src.filter import Filter
from src.workload import run_scenario, get_all_filters, get_scenario_config
from src.multitask_formula import create_model_registry_from_filters
from src import log
import argparse
import os
import sys
import random
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="serval", help="Mode to run: serval or earthsight")
    parser.add_argument("--scenario", default="naturaldisaster", help="Scenario type: naturaldisaster, intelligence, or combined")
    parser.add_argument("--learning", default="mtl", help="mtl or stl")
    parser.add_argument("--hardware", default="tpu", help="Hardware to run on: tpu or cpu")
    parser.add_argument("--hours", type=float, default=24, help="Simulation length in hours (default: 24)")
    args = parser.parse_args()

    random.seed(42)

    args.hardware = "tpu" if args.hardware.lower() in ["tpu", "edgetpu", "edge tpu", "coral"] else "gpu"

    # Build parameterized directory paths for parallel-safe logging and caching
    run_tag = f"{args.hardware}-{args.scenario}-{args.mode}-{args.learning}-{int(args.hours)}h"
    log_dir = os.path.join("logs", f"log_{run_tag}")
    cache_dir = os.path.join("cache", f"0_{run_tag}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    log.reconfigure(os.path.join(log_dir, "log"))
    stdout_log = open(os.path.join(log_dir, "stdout.log"), "w")
    sys.stdout = TeeStream(stdout_log, sys.__stdout__)

    log_config = {
        "mode": args.mode,
        "scenario": args.scenario,
        "learning": args.learning,
        "hardware": args.hardware,
        "hours": args.hours
    }
    
    print("Running on", "Edge TPU" if args.hardware == "tpu" else "GPU")
    print("Running simulations for {} scenario in mode {} with {} learning on {}".format(args.scenario, args.mode, args.learning, args.hardware))
    
    # Load and register filters
    all_filters = get_all_filters(hardware=args.hardware)
    Filter.add_filters(all_filters)
    for filter in all_filters:
        filter.false_negative_rate = 0.05 if args.learning == "mtl" and args.mode == 'earthsight' else 0.035
    registry = create_model_registry_from_filters(all_filters)[0] if args.learning == "mtl" else None

    # Load scenario queries
    queries = run_scenario(args.scenario)
    scenario_config = get_scenario_config(args.scenario)
    workload_intensity = scenario_config.get("workload_intensity", 1)

    # Load satellites and ground stations
    stations = pd.read_json("referenceData/planet_stations.json")
    groundStations: 'List[Station]' = []
    satellites = Satellite.load_from_tle("referenceData/planet_tles.txt")
    satellites = [EarthsightSatellite(i, mode=args.mode, mtl_registry=registry) for i in satellites]

    for satellite in satellites:
        satellite.compute_power = JETSON_POWER_DRAW if args.hardware == 'gpu' else CORAL_POWER_DRAW # Set compute power for hardware options

    # Initialize time and simulator
    startTime = Time().from_str("2025-02-01 14:00:00")
    endTime = startTime.copy()
    endTime.add_seconds(args.hours * 3600)
    sim = Simulator(60, startTime, endTime, satellites, groundStations)

    # Link scheduler to ground stations
    scheduler = EarthSightScheduler(queries, satellites, groundStations, sim.time,
                                    cache_dir=cache_dir)

    for id, row in stations.iterrows():
        s = Station(row["name"], id, Location().from_lat_long(row["location"][0], row["location"][1]))
        groundStations.append(EarthSightGroundStation(s, scheduler=scheduler, mode=get_mode_int(log_config)))   

    # Eun simulation
    Metrics.metr()
    sim.run()

    assess_gs_logs(EarthSightGroundStation.rcv_data.items(), log_dir, log_config)

    # powerjson
    power_data = {
        "power_generation": EarthsightSatellite.power_generation,
        "power_consumptions": EarthsightSatellite.power_consumptions,
        "compute_seconds": EarthsightSatellite.compute_s,
        "total_compute_power": EarthsightSatellite.total_compute_power,
        "pct_of_gen_for_cmpt": EarthsightSatellite.total_compute_power / EarthsightSatellite.power_generation if EarthsightSatellite.power_generation > 0 else 0
    }
    with open(os.path.join(log_dir, "power.json"), "w") as f:
        json.dump(power_data, f, indent=4)

    sys.stdout = sys.__stdout__
    stdout_log.close()
