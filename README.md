# EarthSight: A Distributed Framework for Low-Latency Satellite Intelligence

[![arXiv](https://img.shields.io/badge/arXiv-2511.10834-b31b1b.svg)](https://arxiv.org/abs/2511.10834)

**Ansel Kaplan Erol, Seungjun Lee, Divya Mahajan**

This repository contains the simulation framework for EarthSight, a distributed runtime system that reduces satellite imagery delivery latency by performing on-board ML inference coordinated with ground-station scheduling. EarthSight reduces average compute time per image by **1.9x** and lowers 90th percentile end-to-end latency from **51 to 21 minutes** compared to baseline systems.

---

## Table of Contents

- [EarthSight: A Distributed Framework for Low-Latency Satellite Intelligence](#earthsight-a-distributed-framework-for-low-latency-satellite-intelligence)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Artifact Evaluation](#artifact-evaluation)
    - [Reproducible Results](#reproducible-results)
    - [System Requirements](#system-requirements)
    - [Environment Setup](#environment-setup)
    - [Path A: SLURM Cluster](#path-a-slurm-cluster)
    - [Path B: Single Machine (no SLURM)](#path-b-single-machine-no-slurm)
    - [Result File Locations](#result-file-locations)
  - [Running Individual Simulations](#running-individual-simulations)
    - [Arguments](#arguments)
    - [Output](#output)
  - [Project Structure](#project-structure)
  - [Citation](#citation)
  - [License](#license)

---

## Overview

Traditional satellite imaging pipelines downlink all captured images before analysis, causing delays of hours or days. EarthSight moves analysis on-board while maintaining coordination with ground stations through three key innovations:

1. **Multi-task inference** — Shared backbone networks on satellites distribute computational costs across multiple vision tasks simultaneously, amortizing feature extraction.

2. **Ground-station query scheduler** — Aggregates user requests, forecasts priorities using a 6-hour lookahead simulation, and allocates processing budgets to incoming satellite imagery.

3. **Dynamic filter ordering** — Uses model selectivity, accuracy, and execution cost to evaluate DNF filter formulas in an order that rejects low-value images early, preserving on-board compute and power resources.

---

## Artifact Evaluation

### Reproducible Results

The artifact reproduces the following results from the paper. Two types of scripts are involved: **standalone** scripts that run a benchmark directly (no simulation data needed), and **post-simulation** scripts that read from simulation log files.

| Result | What it shows | Script | Needs simulations? | Wall-clock time |
|--------|---------------|--------|--------------------|-----------------|
| **Table 4** | Per-image compute time for Serval vs. EarthSight STL/MTL; demonstrates the 1.9x compute speedup from dynamic filter ordering | `generate_table_4.py` | No | ~20 min |
| **Table 5** | EarthSight MTL compute time vs. the optimal (oracle) multitask schedule; validates that MTL evaluation is near-optimal | `generate_table_5.py` | No | ~90 min |
| **Table 6** | On-board power consumption breakdown across hardware platforms and scenarios; shows EarthSight stays within power budget | `generate_table_6.py` | Yes | Seconds |
| **Main figure** | 90th percentile end-to-end latency bar chart across all scenarios and hardware (the primary paper result) | `generate_main_result.py` | Yes | Seconds |

**Start with Tables 4 and 5** — they can run immediately without any simulation data.

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
git clone https://github.com/your-org/EarthSight-MLSys2026
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

### Path A: SLURM Cluster

Use this path if you have access to a SLURM-managed HPC cluster. All simulation jobs can run in parallel, so the total wall-clock time is just ~12 hours regardless of how many simulations you submit.

**Step 1 — Generate the .sbatch scripts**

Run from `Sat_Simulator/`:

```bash
# Full suite (18 simulation jobs)
python generate_slurm_scripts.py \
    --cluster-path /absolute/path/to/EarthSight-MLSys2026 \
    --account  YOUR_SLURM_ACCOUNT \
    --email    you@institution.edu

# Resource-limited clusters: combined scenario only (6 simulation jobs)
python generate_slurm_scripts.py \
    --cluster-path /absolute/path/to/EarthSight-MLSys2026 \
    --account  YOUR_SLURM_ACCOUNT \
    --email    you@institution.edu \
    --combined-only
```

`--cluster-path` must be the absolute path to the repository **on the cluster** — it is embedded verbatim in the generated `.sbatch` files. If `--account` or `--email` are omitted, placeholder values are written; edit the `.sbatch` files before submitting.

All scripts are written to `Sat_Simulator/batch_scripts/`.

**Step 2 — Submit jobs**

```bash
cd Sat_Simulator/batch_scripts

# Submit Tables 4 & 5 immediately (standalone — no simulation data needed)
bash launch_tables_standalone.sh

# Submit all simulation jobs (run in parallel on the cluster)
bash launch_sims.sh

# After ALL simulation jobs finish, submit Table 6 and the main result figure
bash launch_tables_postrun.sh
```

**Generated files reference**

| File | Purpose | SLURM resources |
|------|---------|-----------------|
| `earthsight-<scenario>-<mode>-<hw>.sbatch` | One simulation per configuration | 256 GB, 14 h limit |
| `generate_table4.sbatch` | Standalone Table 4 | 64 GB, 30 min limit |
| `generate_table5.sbatch` | Standalone Table 5 | 64 GB, 2 h limit |
| `generate_table6.sbatch` | Post-sim Table 6 | 32 GB, 15 min limit |
| `generate_main_result.sbatch` | Post-sim main figure | 32 GB, 15 min limit |
| `launch_sims.sh` | Submits all simulation `.sbatch` files | — |
| `launch_tables_standalone.sh` | Submits Tables 4 & 5 | — |
| `launch_tables_postrun.sh` | Submits Table 6 & main result | — |

---

### Path B: Single Machine (no SLURM)

Use this path if you do not have SLURM access. Simulations run serially, one at a time.

> **Memory:** 256 GB RAM is required.
> **Windows users:** Shell scripts require a bash-compatible environment. Use WSL (Windows Subsystem for Linux) or Git Bash.

**Step 1 — Generate the shell scripts** (run once from `Sat_Simulator/`)

```bash
# Full suite — generates 18 individual scripts + 6 group scripts + all helpers
python generate_batch_scripts.py

# Combined scenario only — generates 6 individual scripts + 2 group scripts + helpers
python generate_batch_scripts.py --combined-only
```

All scripts are written to `Sat_Simulator/batch_scripts/`.

> **Note:** If the generated scripts are not executable, make them so before running:
> ```bash
> chmod +x batch_scripts/*.sh batch_scripts/individual/*.sh
> ```
> Alternatively, prefix every command below with `bash` (e.g. `bash batch_scripts/run_combined_tpu.sh`).

**Step 2 — Run standalone tables immediately** (no simulation data needed)

```bash
bash batch_scripts/generate_table_4.sh   # ~20 min  -> results/table4.txt
bash batch_scripts/generate_table_5.sh   # ~90 min  -> results/table5.txt
```

**Step 3 — Run simulations**

Choose the scope that fits your available time and hardware:

```bash
# Option A: All-in-one pipeline helpers (recommended)
# Tables 4 & 5 run in the background while simulations proceed.
bash batch_scripts/run_all_combined.sh   # combined scenario only (~3 days)
bash batch_scripts/run_all_full.sh       # all scenarios           (~9 days)

# Option B: Combined-scenario convenience scripts
bash batch_scripts/run_combined_tpu.sh   # combined + TPU, all 3 modes  (~36 h)
bash batch_scripts/run_combined_gpu.sh   # combined + GPU, all 3 modes  (~36 h)
bash batch_scripts/run_combined_both.sh  # both combined groups          (~72 h)

# Option C: Individual simulation scripts (one per run, in batch_scripts/individual/)
bash batch_scripts/individual/run_combined_serval_tpu.sh
bash batch_scripts/individual/run_combined_stl_tpu.sh
bash batch_scripts/individual/run_combined_mtl_tpu.sh
# ... (18 scripts total, one per scenario x mode x hardware combination)
```

**Step 4 — Generate remaining results** (after all simulations finish)

```bash
bash batch_scripts/generate_results_postrun.sh
# Produces: results/table6.txt  and  results/summary_plot.png
```

**Generated script structure**

```
batch_scripts/
├── individual/                        Individual simulation scripts (18 total)
│   ├── run_combined_serval_tpu.sh     ~12 h each
│   ├── run_combined_stl_tpu.sh
│   ├── run_combined_mtl_tpu.sh
│   ├── run_combined_serval_gpu.sh
│   └── ...                            (one per scenario x mode x hardware)
├── run_combined_tpu.sh                Group: 3 modes for combined + TPU    (~36 h)
├── run_combined_gpu.sh                Group: 3 modes for combined + GPU    (~36 h)
├── run_combined_both.sh               Combined TPU then GPU                (~72 h)
├── run_naturaldisaster_tpu.sh         Group scripts for other scenarios
├── run_naturaldisaster_gpu.sh
├── run_intelligence_tpu.sh
├── run_intelligence_gpu.sh
├── run_all_combined.sh                Full pipeline, combined only         (~3 days)
├── run_all_full.sh                    Full pipeline, all scenarios         (~9 days)
├── generate_table_4.sh                Standalone (~20 min)
├── generate_table_5.sh                Standalone (~90 min)
└── generate_results_postrun.sh        Post-sim (seconds)
```

---

### Result File Locations

All result files are written to `Sat_Simulator/results/` (created automatically).

| File | Content | Generated by |
|------|---------|--------------|
| `results/table4.txt` | Table 4 — per-image compute time comparison | `generate_table_4.py` |
| `results/table5.txt` | Table 5 — MTL vs. oracle compute time | `generate_table_5.py` |
| `results/table6.txt` | Table 6 — power consumption analysis | `generate_table_6.py` |
| `results/summary_plot.png` | Main result figure — 90th percentile latency | `generate_main_result.py` |

Simulation log files are stored in `Sat_Simulator/logs/log_<run-tag>/` and are read by the post-simulation scripts.

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
