#!/usr/bin/env python3
"""
generate_slurm_scripts.py -- Generate SLURM batch scripts for the EarthSight artifact.

Overview
--------
This script creates one .sbatch file per simulation configuration, plus dedicated
scripts for standalone table generation (Tables 4 & 5) and post-simulation result
generation (Table 6 and the main result figure).

Each simulation runs 48 simulated hours of satellite operation, which takes
approximately 12 hours of wall-clock time. SLURM jobs are allocated 14 hours
(12 h expected + 2 h buffer). Memory requirement is 256 GB per simulation job.

Simulation matrix
-----------------
  Modes     : serval | earthsight (STL) | earthsight (MTL)
  Scenarios : combined | intelligence | naturaldisaster
  Hardware  : tpu | gpu

  Full suite  : 18 simulations -- jobs run in parallel on the cluster
  Combined only: 6 simulations -- use --combined-only when resources are limited

Outputs (written to Sat_Simulator/batch_scripts/)
--------------------------------------------------
  Simulation jobs (one per combination):
    earthsight-<scenario>-<mode>-<hw>.sbatch
      #SBATCH --time=14:00:00   (~12 h expected + 2 h buffer)
      #SBATCH --mem=256G

  Standalone table jobs (no prior simulation needed):
    generate_table4.sbatch    (~20 min,  30 min SLURM limit)
    generate_table5.sbatch    (~90 min,  2 h SLURM limit)

  Post-simulation jobs (run AFTER all simulations complete):
    generate_table6.sbatch         (seconds, 15 min SLURM limit)
    generate_main_result.sbatch    (seconds, 15 min SLURM limit)

  Launch helpers:
    launch_sims.sh              -- sbatch all simulation jobs
    launch_tables_standalone.sh -- sbatch Tables 4 & 5 (no deps, submit anytime)
    launch_tables_postrun.sh    -- sbatch Table 6 & main result (after sims)

Usage
-----
  python generate_slurm_scripts.py \\
      --cluster-path /path/to/EarthSight-MLSys2026 \\
      --account  gts-xyz123 \\
      --email    you@university.edu \\
      [--combined-only]

  Then submit:
    cd Sat_Simulator/batch_scripts
    bash launch_tables_standalone.sh   # submit Tables 4 & 5 immediately
    bash launch_sims.sh                # submit all simulation jobs
    # ... wait for all simulation jobs to finish ...
    bash launch_tables_postrun.sh      # submit Table 6 & main result
"""

import os
import argparse
from itertools import product


# ---------------------------------------------------------------------------
# Simulation constants (not user-configurable)
# ---------------------------------------------------------------------------

# Simulated scenario duration passed to run.py. Fixed at 48 h for reproducibility.
SIMULATION_HOURS = 48

# SLURM wall-clock limit for simulation jobs (expected ~12 h + 2 h buffer).
SLURM_SIM_WALL   = "14:00:00"

# Each entry is (short_label, run.py --mode argument).
MODES = [
    ("serval", "serval"),
    ("stl",    "earthsight --learning stl"),
    ("mtl",    "earthsight --learning mtl"),
]

ALL_SCENARIOS = ["combined", "intelligence", "naturaldisaster"]
HARDWARE      = ["tpu", "gpu"]


# ---------------------------------------------------------------------------
# Script template builders
# ---------------------------------------------------------------------------

def _sim_sbatch(job_name: str, command: str, account: str, email: str,
                repo_path: str) -> str:
    """Return the contents of a simulation .sbatch script."""
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --account={account}
#SBATCH --time={SLURM_SIM_WALL}
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

set -euo pipefail

source {repo_path}/satsim/bin/activate
cd {repo_path}/Sat_Simulator/

{command}
"""


def _table_sbatch(job_name: str, command: str, account: str, email: str,
                  wall_time: str, mem: str, repo_path: str) -> str:
    """Return the contents of a table-generation .sbatch script."""
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --mem={mem}
#SBATCH --account={account}
#SBATCH --time={wall_time}
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={email}

set -euo pipefail

source {repo_path}/satsim/bin/activate
cd {repo_path}/Sat_Simulator/

{command}
"""


def _write_launch_script(path: str, files: list, comment: str) -> None:
    """Write a launch helper that sbatch-es each listed .sbatch file."""
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"# {comment}",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        "",
    ]
    for f in files:
        lines.append(f'sbatch "$SCRIPT_DIR/{f}"')
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cluster-path", required=True, metavar="PATH",
        help="Absolute path to the EarthSight-MLSys2026 repo root on the cluster "
             "(e.g. /home/user/scratch/EarthSight-MLSys2026). "
             "This value is embedded verbatim in every generated script.",
    )
    parser.add_argument(
        "--account", default="PLACEHOLDER_ACCOUNT", metavar="ACCT",
        help="SLURM account name (e.g. gts-xyz123). "
             "Defaults to 'PLACEHOLDER_ACCOUNT' -- edit the generated scripts if omitted.",
    )
    parser.add_argument(
        "--email", default="PLACEHOLDER_EMAIL", metavar="EMAIL",
        help="Email address for job completion/failure notifications. "
             "Defaults to 'PLACEHOLDER_EMAIL' -- edit the generated scripts if omitted.",
    )
    parser.add_argument(
        "--combined-only", action="store_true",
        help="Generate simulation scripts for the 'combined' scenario only "
             "(6 jobs instead of 18). Recommended when cluster resources are limited. "
             "Sufficient to validate the core latency claims.",
    )

    args = parser.parse_args()

    repo_path = args.cluster_path.rstrip("/")
    scenarios = ["combined"] if args.combined_only else ALL_SCENARIOS

    here       = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(here, "batch_scripts")
    logs_dir   = os.path.join(here, "logs")
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    sim_scripts = []

    # ------------------------------------------------------------------
    # 1. Per-simulation sbatch scripts
    # ------------------------------------------------------------------
    print("Generating simulation scripts...")
    for scenario, (mode_label, mode_arg), hw in product(scenarios, MODES, HARDWARE):
        job_name = f"earthsight-{scenario}-{mode_label}-{hw}"
        filename = f"{job_name}.sbatch"
        path     = os.path.join(script_dir, filename)

        command = (
            f"python run.py "
            f"--mode {mode_arg} "
            f"--scenario {scenario} "
            f"--hardware {hw} "
            f"--hours {SIMULATION_HOURS}"
        )
        contents = _sim_sbatch(job_name, command, args.account, args.email, repo_path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(contents)

        sim_scripts.append(filename)
        print(f"  {filename}  (~12 h wall-clock, 256 GB)")

    # ------------------------------------------------------------------
    # 2. Standalone table scripts  (no prior simulation needed)
    # ------------------------------------------------------------------
    print("\nGenerating standalone table scripts...")

    with open(os.path.join(script_dir, "generate_table4.sbatch"), "w", encoding="utf-8") as fh:
        fh.write(_table_sbatch(
            job_name  = "earthsight-table4",
            command   = "python generate_table_4.py",
            account   = args.account,
            email     = args.email,
            wall_time = "0:30:00",
            mem       = "64G",
            repo_path = repo_path,
        ))
    print("  generate_table4.sbatch  (~20 min wall-clock, no simulation data needed)")

    with open(os.path.join(script_dir, "generate_table5.sbatch"), "w", encoding="utf-8") as fh:
        fh.write(_table_sbatch(
            job_name  = "earthsight-table5",
            command   = "python generate_table_5.py",
            account   = args.account,
            email     = args.email,
            wall_time = "2:00:00",
            mem       = "64G",
            repo_path = repo_path,
        ))
    print("  generate_table5.sbatch  (~90 min wall-clock, no simulation data needed)")

    # ------------------------------------------------------------------
    # 3. Post-simulation result scripts  (require completed simulation logs)
    # ------------------------------------------------------------------
    print("\nGenerating post-simulation scripts...")

    with open(os.path.join(script_dir, "generate_table6.sbatch"), "w", encoding="utf-8") as fh:
        fh.write(_table_sbatch(
            job_name  = "earthsight-table6",
            command   = "mkdir -p results\npython generate_table_6.py | tee results/table6.txt",
            account   = args.account,
            email     = args.email,
            wall_time = "0:15:00",
            mem       = "32G",
            repo_path = repo_path,
        ))
    print("  generate_table6.sbatch  (seconds; requires simulation logs in logs/)")

    with open(os.path.join(script_dir, "generate_main_result.sbatch"), "w", encoding="utf-8") as fh:
        fh.write(_table_sbatch(
            job_name  = "earthsight-main-result",
            command   = "python generate_main_result.py",
            account   = args.account,
            email     = args.email,
            wall_time = "0:15:00",
            mem       = "32G",
            repo_path = repo_path,
        ))
    print("  generate_main_result.sbatch  (seconds; requires simulation logs in logs/)")

    # ------------------------------------------------------------------
    # 4. Launch helpers
    # ------------------------------------------------------------------
    print("\nGenerating launch helpers...")

    _write_launch_script(
        path    = os.path.join(script_dir, "launch_sims.sh"),
        files   = sim_scripts,
        comment = "Submit all simulation jobs to SLURM. Jobs run in parallel.",
    )
    print("  launch_sims.sh")

    _write_launch_script(
        path    = os.path.join(script_dir, "launch_tables_standalone.sh"),
        files   = ["generate_table4.sbatch", "generate_table5.sbatch"],
        comment = "Submit standalone table jobs (Tables 4 & 5). No deps -- submit anytime.",
    )
    print("  launch_tables_standalone.sh")

    _write_launch_script(
        path    = os.path.join(script_dir, "launch_tables_postrun.sh"),
        files   = ["generate_table6.sbatch", "generate_main_result.sbatch"],
        comment = "Submit post-simulation jobs. Run ONLY after all simulation jobs finish.",
    )
    print("  launch_tables_postrun.sh")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_sims = len(sim_scripts)
    scope  = "combined scenario only" if args.combined_only else "all scenarios"
    print(f"\n{'='*60}")
    print(f"Generated {n_sims} simulation scripts ({scope}) + 4 table scripts")
    print(f"Wall-clock / sim job : ~12 h  (SLURM limit: {SLURM_SIM_WALL})")
    print(f"Memory / sim job     : 256 GB")
    print(f"Simulated hours      : {SIMULATION_HOURS} h per run (fixed)")
    print(f"Output directory     : {script_dir}")
    if args.account == "PLACEHOLDER_ACCOUNT" or args.email == "PLACEHOLDER_EMAIL":
        print("\nWARNING: --account or --email was not provided.")
        print("  Edit the #SBATCH lines in the generated .sbatch files before submitting.")
    print(f"\nNext steps:")
    print(f"  cd Sat_Simulator/batch_scripts")
    print(f"  bash launch_tables_standalone.sh   # submit Tables 4 & 5 now")
    print(f"  bash launch_sims.sh                # submit all {n_sims} simulation jobs")
    print(f"  # ... wait for all simulation jobs to finish ...")
    print(f"  bash launch_tables_postrun.sh      # submit Table 6 & main result")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
