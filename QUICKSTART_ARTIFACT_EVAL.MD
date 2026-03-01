# EarthSight -- Artifact Evaluation Quickstart

For complete documentation, see [README.md](README.md).

---

## Before You Begin

| Requirement | Value |
|-------------|-------|
| RAM | **256 GB** (hard requirement for simulation runs) |
| Python | 3.9 or 3.13 |
| Wall-clock time per simulation | ~12 h (48 simulated hours) |
| Serial total -- combined scenario (6 sims) | ~3 days |
| Serial total -- full suite (18 sims) | ~9 days |
| SLURM cluster | Jobs run in parallel; ~12 h total regardless of suite size |

---

## 1. Environment Setup

```bash
python -m venv ./satsim

# Linux/macOS
source satsim/bin/activate

# Windows (PowerShell)
satsim\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 2. What Gets Reproduced

| Result | Standalone? | Time |
|--------|------------|------|
| **Table 4** — Per-image compute time (Serval vs. STL vs. MTL); shows 1.9x compute speedup | Yes | ~20 min |
| **Table 5** — EarthSight MTL vs. optimal oracle; validates near-optimal scheduling | Yes | ~90 min |
| **Table 6** — On-board power consumption across hardware/scenarios | No (needs sims) | Seconds |
| **Main figure** — 90th percentile end-to-end latency across all scenarios and hardware | No (needs sims) | Seconds |

**Start with Tables 4 & 5** -- they run immediately with no simulation data.

---

## 3. Tables 4 & 5 (standalone, run immediately)

```bash
cd Sat_Simulator
python generate_table_4.py   # -> results/table4.txt  (~20 min)
python generate_table_5.py   # -> results/table5.txt  (~90 min)
```

---

## 4. Table 6 & Main Figure (require simulation data)

Choose the path that matches your setup.

---

### Option A -- SLURM Cluster

Jobs run in parallel; expect ~12 h of wall time regardless of suite size.

```bash
cd Sat_Simulator

# Generate .sbatch scripts
python generate_slurm_scripts.py \
    --cluster-path /absolute/path/to/EarthSight-MLSys2026 \
    --account  YOUR_SLURM_ACCOUNT \
    --email    you@institution.edu \
    --combined-only        # remove for full 18-simulation suite

cd batch_scripts

# Submit standalone tables immediately (no simulation data needed)
bash launch_tables_standalone.sh

# Submit simulation jobs (run in parallel on the cluster)
bash launch_sims.sh

# After ALL simulation jobs finish:
bash launch_tables_postrun.sh   # Table 6 + main result figure
```

---

### Option B -- Single Machine (no SLURM)

Requires 256 GB RAM. Simulations run serially.
**Windows:** use WSL or Git Bash for the `.sh` scripts.

```bash
cd Sat_Simulator

# Generate shell scripts
python generate_batch_scripts.py --combined-only
# (remove --combined-only for the full 18-simulation suite)

# Make scripts executable (or prefix every command below with 'bash')
chmod +x batch_scripts/*.sh batch_scripts/individual/*.sh

# Run standalone tables immediately (background-friendly):
bash batch_scripts/generate_table_4.sh   # ~20 min
bash batch_scripts/generate_table_5.sh   # ~90 min

# Run simulations -- choose one:
bash batch_scripts/run_all_combined.sh   # combined scenario, all groups  (~3 days)
bash batch_scripts/run_combined_tpu.sh   # combined + TPU only            (~36 h)
bash batch_scripts/run_combined_gpu.sh   # combined + GPU only            (~36 h)
bash batch_scripts/run_combined_both.sh  # combined TPU then GPU          (~72 h)

# Or run a single simulation at a time (batch_scripts/individual/):
bash batch_scripts/individual/run_combined_serval_tpu.sh   # ~12 h
bash batch_scripts/individual/run_combined_stl_tpu.sh      # ~12 h
bash batch_scripts/individual/run_combined_mtl_tpu.sh      # ~12 h
# ... (18 scripts total)

# After all desired simulations finish:
bash batch_scripts/generate_results_postrun.sh
```

---

## 5. Results

| File | Content |
|------|---------|
| `Sat_Simulator/results/table4.txt` | Table 4 |
| `Sat_Simulator/results/table5.txt` | Table 5 |
| `Sat_Simulator/results/table6.txt` | Table 6 |
| `Sat_Simulator/results/summary_plot.png` | Main result figure |
