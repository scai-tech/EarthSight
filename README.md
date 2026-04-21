# EarthSight: A Distributed Framework for Low-Latency Satellite Intelligence

[![arXiv](https://img.shields.io/badge/arXiv-2511.10834-b31b1b.svg)](https://arxiv.org/abs/2511.10834)

**Ansel Kaplan Erol, Seungjun Lee, Divya Mahajan**

*EarthSight* has been accepted to **MLSys 2026!** Please click the Arxiv tile above to view our pre-print.

This repository contains the simulation framework for EarthSight, a distributed runtime system that reduces satellite imagery delivery latency by performing on-board ML inference coordinated with ground-station scheduling. EarthSight reduces average compute time per image by **1.9x** and lowers 90th percentile end-to-end latency from **51 to 21 minutes** compared to baseline systems.

<div align="center">
  <img src="https://github.com/user-attachments/assets/91cb12f4-4b1e-4b06-af77-cdce394132c9" alt="image" width="370" height="322" />
</div>
---

## Overview

Traditional satellite imaging pipelines downlink all captured images before analysis, causing delays of hours or days. EarthSight moves analysis on-board while maintaining coordination with ground stations through three key innovations:

1. **Multi-task inference** — Shared backbone networks on satellites distribute computational costs across multiple vision tasks simultaneously, amortizing feature extraction.

2. **Ground-station query scheduler** — Aggregates user requests, forecasts priorities using a 6-hour lookahead simulation, and allocates processing budgets to incoming satellite imagery.

3. **Dynamic filter ordering** — Uses model selectivity, accuracy, and execution cost to evaluate DNF filter formulas in an order that rejects low-value images early, preserving on-board compute and power resources.

<div align="center">
  <img src="https://github.com/user-attachments/assets/fc697340-0ff9-4ab8-94bc-e6e86d986a90" alt="image" width="569" height="304" />
</div>
---

## Artifact Evaluation

*EarthSight* has been awarded the artifacts available and artifacts functional badges. Certain simulation results were unable to be reproduced during the review period due to memory requirements and run-time duration.

### Reproducible Results

The artifact reproduces the following results from the paper. Two types of scripts are involved: **standalone** scripts that run a benchmark directly (no simulation data needed), and **post-simulation** scripts that read from simulation log files.

| Result | What it shows | Script | Needs simulations? | Wall-clock time |
|--------|---------------|--------|--------------------|-----------------|
| **Table 4** | Per-image compute time for Serval vs. EarthSight STL/MTL; demonstrates the 1.9x compute speedup from dynamic filter ordering | `generate_table_4.py` | No | ~20 min |
| **Table 5** | EarthSight MTL compute time vs. the optimal (oracle) multitask schedule; validates that MTL evaluation is near-optimal | `generate_table_5.py` | No | ~90 min |
| **Table 3** | On-board power consumption breakdown across hardware platforms and scenarios; shows EarthSight stays within power budget | `generate_table_3.py` | Yes | Seconds |
| **Main figure** | 90th percentile end-to-end latency bar chart across all scenarios and hardware (the primary paper result) | `generate_main_result.py` | Yes | Seconds |

---

### System Requirements

| Requirement | Value |
|-------------|-------|
| **RAM** | 256 GB (hard requirement for simulation jobs) |
| **Python** | 3.9 or 3.13 |
| **OS** | Linux or macOS (Windows: use WSL or Git Bash for shell scripts) |
| **Wall-clock time per simulation** | ~12 hours (48 simulated hours of satellite operation) |
| **Full suite (18 simulations, serial)** | ~9 days |
| **Combined scenario only (6 simulations, serial)** | ~3 days |
| **On a SLURM cluster** | Simulations run in parallel; total wall time ~12 h regardless of suite size |

> **Recommended scope for resource-constrained evaluation:** Run the **combined scenario only** (6 simulations). This covers the Urban Observation rows in the main figure and is sufficient to validate the core latency claims.

---

### Environment Setup

All artifact scripts are run from the `Sat_Simulator/` directory unless otherwise noted.

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/scai-tech/EarthSight-MLSys2026
cd EarthSight-MLSys2026

# 2. Create and activate a Python virtual environment
python -m venv ./satsim

# Linux/macOS
source satsim/bin/activate

# Windows (PowerShell)
satsim\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running Individual Simulations

Simulations can also be run directly from `Sat_Simulator/`:

```bash
python run.py --mode earthsight --scenario naturaldisaster --learning mtl --hardware tpu
```

Each simulation runs **48 simulated hours** of satellite operation (approximately **12 hours of wall-clock time**) and requires **256 GB of RAM**.

### Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--mode` | `serval`, `earthsight` | Scheduling algorithm: Serval (baseline) or EarthSight (ML-augmented) |
| `--scenario` | `naturaldisaster`, `intelligence`, `combined`, `coverage_scaling` | Query scenario defining geographic areas of interest and priorities |
| `--learning` | `mtl`, `stl` | Multitask learning (shared backbones) vs. single-task learning |
| `--hardware` | `tpu`, `gpu` | Target hardware: Edge TPU @ 2W (Coral) or GPU @ 30W (Jetson) |

The `--hours` argument (simulated hours) is fixed at 48 in all artifact scripts and should not be changed.

### Output

Simulation results are saved to `Sat_Simulator/logs/log_<hardware>-<scenario>-<mode>-<learning>-48h/`:

| File | Content |
|------|---------|
| `stdout.log` | Full console output with summary statistics |
| `summary.json` | Latency metrics by priority tier; read by `generate_main_result.py` |
| `power.json` | Power generation and consumption data; read by `generate_table_6.py` |

---

## Project Structure

```
EarthSight-MLSys2026/
├── requirements.txt
├── Sat_Simulator/
│   ├── run.py                        Simulation entry point
│   ├── generate_slurm_scripts.py     Generate SLURM .sbatch files
│   ├── generate_batch_scripts.py     Generate serial shell scripts (no SLURM)
│   ├── generate_table_4.py           Standalone: compute time benchmark
│   ├── generate_table_5.py           Standalone: multitask oracle benchmark
│   ├── generate_table_6.py           Post-sim: power consumption analysis
│   ├── generate_main_result.py       Post-sim: 90th-pct latency bar chart
│   ├── batch_scripts/                Generated scripts (created at runtime)
│   │   └── individual/               One script per simulation run
│   ├── logs/                         Simulation log files (created at runtime)
│   ├── results/                      Tables and figures (created at runtime)
│   ├── src/
│   │   ├── simulator.py              Main simulation loop (60 s timesteps)
│   │   ├── earthsightsatellite.py    On-board image capture, ML eval, power mgmt
│   │   ├── earthsightgs.py           Ground station: scheduling & delay tracking
│   │   ├── scheduler.py              Query-to-schedule pipeline with lookahead
│   │   ├── formula.py                STL DNF formula evaluation & filter ordering
│   │   ├── multitask_formula.py      MTL evaluation with shared backbone models
│   │   ├── image.py                  Image representation & evaluation dispatcher
│   │   ├── query.py                  Spatial queries with R-tree indexing
│   │   ├── filter.py                 ML filter registry (pass probs, exec times)
│   │   ├── satellite.py              Orbital mechanics via Skyfield TLEs
│   │   ├── topology.py               Satellite-ground station visibility
│   │   ├── routing.py                Downlink scheduling
│   │   ├── transmission.py           Packet transfer with SNR/PER modeling
│   │   └── ...
│   └── referenceData/
│       ├── planet_tles.txt           Planet Labs satellite TLEs
│       ├── planet_stations.json      Ground station locations
│       ├── de440s.bsp                JPL ephemeris (32 MB)
│       └── scenarios.py              Scenario definitions (regions, queries, filters)
```

---

## Citation

```bibtex
@article{erol2025earthsight,
  title={EarthSight: A Distributed Framework for Low-Latency Satellite Intelligence},
  author={Erol, Ansel Kaplan and Lee, Seungjun and Mahajan, Divya},
  journal={arXiv preprint arXiv:2511.10834},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
