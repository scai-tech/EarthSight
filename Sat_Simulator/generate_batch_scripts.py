#!/usr/bin/env python3
"""
generate_batch_scripts.py -- Generate serial shell scripts for EarthSight artifact
evaluation on machines without SLURM.

Overview
--------
Each simulation runs 48 simulated hours of satellite operation, which takes
approximately 12 hours of wall-clock time. The full suite is 18 simulations
(3 modes x 3 scenarios x 2 hardware targets); running them serially takes
roughly 9 days. The combined-scenario subset (6 simulations) takes roughly 3 days.

Memory requirement: 256 GB RAM is required to run simulations.

Generated scripts (written to Sat_Simulator/batch_scripts/)
------------------------------------------------------------
  Individual simulation scripts (batch_scripts/individual/):
    run_<scenario>_<mode>_<hw>.sh     one per simulation run (18 total)

  Group scripts -- one (scenario, hardware) pair per script, all 3 modes in serial:
    run_combined_tpu.sh               3 sims, ~36 h wall-clock
    run_combined_gpu.sh               3 sims, ~36 h wall-clock
    run_naturaldisaster_tpu.sh        3 sims, ~36 h wall-clock  [full suite]
    run_naturaldisaster_gpu.sh        3 sims, ~36 h wall-clock  [full suite]
    run_intelligence_tpu.sh           3 sims, ~36 h wall-clock  [full suite]
    run_intelligence_gpu.sh           3 sims, ~36 h wall-clock  [full suite]

  Combined-scenario convenience scripts:
    run_combined_tpu.sh               (same as group script above)
    run_combined_gpu.sh               (same as group script above)
    run_combined_both.sh              combined TPU then GPU, ~72 h wall-clock total

  Standalone tables (no simulation data needed):
    generate_table_4.sh               ~20 min
    generate_table_5.sh               ~90 min

  Post-simulation results (run after all simulations finish):
    generate_results_postrun.sh       seconds

  All-in-one pipeline helpers:
    run_all_combined.sh               full pipeline, combined scenario only  (~3 days)
    run_all_full.sh                   full pipeline, all scenarios            (~9 days)

Usage
-----
  # Step 1 -- generate the scripts (run once):
  cd Sat_Simulator
  python generate_batch_scripts.py [--combined-only]

  # Step 2 -- run standalone tables immediately (no simulation data needed):
  bash batch_scripts/generate_table_4.sh   # ~20 min
  bash batch_scripts/generate_table_5.sh   # ~90 min

  # Step 3 -- run simulations.  Choose one:
  bash batch_scripts/run_all_combined.sh      # combined scenario only (~3 days)
  bash batch_scripts/run_all_full.sh          # full suite (~9 days)

  # Or run individual groups manually:
  bash batch_scripts/run_combined_tpu.sh      # ~36 h
  bash batch_scripts/run_combined_gpu.sh      # ~36 h
  bash batch_scripts/run_combined_both.sh     # both combined groups, ~72 h

  # Step 4 -- after all simulations finish, generate Table 6 and the main figure:
  bash batch_scripts/generate_results_postrun.sh

Options
-------
  --combined-only   Generate scripts for the 'combined' scenario only.
                    This produces 6 individual scripts and the combined group/helper
                    scripts. Sufficient to validate the core latency claims.
                    Reduces total simulation time from ~9 days to ~3 days.

Platform notes
--------------
  Linux/macOS: Run scripts directly with bash.
  Windows    : Use WSL (Windows Subsystem for Linux) or Git Bash.
               Example in WSL: bash batch_scripts/run_combined_tpu.sh
"""

import os
import argparse
from itertools import product


# ---------------------------------------------------------------------------
# Simulation constants (not user-configurable)
# ---------------------------------------------------------------------------

# Simulated scenario duration passed to run.py. Fixed for reproducibility.
SIMULATION_HOURS   = 48

# Approximate wall-clock time per individual simulation run.
WALL_CLOCK_H       = 12

# Each entry is (short_label, full run.py --mode argument).
MODES = [
    ("serval", "serval"),
    ("stl",    "earthsight --learning stl"),
    ("mtl",    "earthsight --learning mtl"),
]

ALL_SCENARIOS = ["combined", "intelligence", "naturaldisaster"]
HARDWARE      = ["tpu", "gpu"]


# ---------------------------------------------------------------------------
# Shell script header template
# ---------------------------------------------------------------------------

_HEADER = """\
#!/usr/bin/env bash
# -----------------------------------------------------------------------
# {title}
# {subtitle}
# -----------------------------------------------------------------------
# Memory requirement: 256 GB RAM
# -----------------------------------------------------------------------
set -euo pipefail

# Resolve paths relative to this script so it can be called from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SIM_DIR="$(cd "$SCRIPT_DIR/{sim_rel}" && pwd)"
REPO_DIR="$(cd "$SIM_DIR/.." && pwd)"

# Activate the Python virtual environment.
source "$REPO_DIR/satsim/bin/activate"
cd "$SIM_DIR"

"""

# For scripts in batch_scripts/ the Sat_Simulator dir is one level up (..)
# For scripts in batch_scripts/individual/ it is two levels up (../..)
_HEADER_ROOT  = _HEADER.replace("{sim_rel}", "..")
_HEADER_INDIV = _HEADER.replace("{sim_rel}", "../..")


# ---------------------------------------------------------------------------
# Script content builders
# ---------------------------------------------------------------------------

def _individual_script(scenario: str, hw: str, mode_label: str,
                        mode_arg: str) -> str:
    """Return content for a single-simulation script."""
    return (
        _HEADER_INDIV.format(
            title    = f"Single run: scenario={scenario}  hardware={hw}  mode={mode_label}",
            subtitle = f"Simulates {SIMULATION_HOURS} h of satellite operation "
                       f"(~{WALL_CLOCK_H} h wall-clock time)",
        )
        + f"echo 'Starting: scenario={scenario}  hardware={hw}  mode={mode_label}'\n"
        + f"echo \"Time: $(date)\"\n"
        + f"python run.py --mode {mode_arg} --scenario {scenario} "
          f"--hardware {hw} --hours {SIMULATION_HOURS}\n"
        + f"echo \"Done. Time: $(date)\"\n"
        + "\n"
    )


def _group_script(scenario: str, hw: str, modes: list) -> str:
    """Return content for a group script running all modes for one (scenario, hw)."""
    n_sims    = len(modes)
    total_h   = n_sims * WALL_CLOCK_H
    mode_list = ", ".join(l for l, _ in modes)
    lines = [
        _HEADER_ROOT.format(
            title    = f"Group: scenario={scenario}   hardware={hw}",
            subtitle = f"Runs {n_sims} simulations serially ({mode_list}) "
                       f"-- est. {total_h} h wall-clock total",
        )
    ]
    for mode_label, mode_arg in modes:
        lines += [
            f"echo '================================================================='",
            f"echo 'Starting: scenario={scenario}  hardware={hw}  mode={mode_label}'",
            f"echo \"Time: $(date)\"",
            f"echo '================================================================='",
            f"python run.py --mode {mode_arg} --scenario {scenario} "
            f"--hardware {hw} --hours {SIMULATION_HOURS}",
            f"echo",
        ]
    lines += [
        "echo '================================================================='",
        "echo 'Group complete.'",
        f"echo \"Time: $(date)\"",
        "",
    ]
    return "\n".join(lines)


def _combined_both_script() -> str:
    """Return content for the combined-both script (combined TPU then GPU)."""
    total_h = 2 * len(MODES) * WALL_CLOCK_H
    lines = [
        _HEADER_ROOT.format(
            title    = "Combined scenario: TPU then GPU (all 6 combined simulations)",
            subtitle = f"Runs {2 * len(MODES)} simulations serially "
                       f"-- est. {total_h} h wall-clock total",
        ),
        'echo "--- Starting combined / TPU group ---"',
        'bash "$SCRIPT_DIR/run_combined_tpu.sh"',
        "",
        'echo "--- Starting combined / GPU group ---"',
        'bash "$SCRIPT_DIR/run_combined_gpu.sh"',
        "",
        "echo '================================================================='",
        "echo 'run_combined_both.sh complete.'",
        "echo \"Time: $(date)\"",
        "",
    ]
    return "\n".join(lines)


def _table_script(py_script: str, description: str, time_est: str) -> str:
    """Return content for a standalone table-generation script."""
    return (
        _HEADER_ROOT.format(
            title    = description,
            subtitle = f"Estimated wall time: {time_est}  (no simulation data needed)",
        )
        + f"echo 'Starting: {description}'\n"
        + f"echo \"Time: $(date)\"\n"
        + f"python {py_script}\n"
        + f"echo \"Done. Time: $(date)\"\n"
        + "\n"
    )


def _postrun_script() -> str:
    """Return content for the post-simulation results script."""
    return (
        _HEADER_ROOT.format(
            title    = "Post-simulation result generation",
            subtitle = "Generates Table 6 and the main result figure. "
                       "Run ONLY after all simulations have finished.",
        )
        + "mkdir -p results\n\n"
        + "echo 'Generating Table 6 (power analysis)...'\n"
        + "python generate_table_6.py | tee results/table6.txt\n\n"
        + "echo 'Generating main result figure...'\n"
        + "python generate_main_result.py\n\n"
        + "echo\n"
        + "echo 'All results written to Sat_Simulator/results/'\n"
        + "echo \"Time: $(date)\"\n"
        + "\n"
    )


def _run_all_script(group_scripts: list) -> str:
    """Return content for an all-in-one pipeline script."""
    n_groups = len(group_scripts)
    n_sims   = n_groups * len(MODES)
    total_h  = n_sims * WALL_CLOCK_H
    desc     = "all scenarios" if n_groups > 2 else "combined scenario only"
    lines = [
        _HEADER_ROOT.format(
            title    = f"Full EarthSight artifact pipeline -- {desc}",
            subtitle = f"{n_sims} simulations serial, est. {total_h} h wall-clock total",
        ),
        "# Tables 4 & 5 run in the background so they overlap with simulations.",
        'echo "Starting standalone table generation in the background..."',
        'mkdir -p "$SIM_DIR/logs"',
        'bash "$SCRIPT_DIR/generate_table_4.sh" > "$SIM_DIR/logs/table4_bg.log" 2>&1 &',
        'TABLE4_PID=$!',
        'bash "$SCRIPT_DIR/generate_table_5.sh" > "$SIM_DIR/logs/table5_bg.log" 2>&1 &',
        'TABLE5_PID=$!',
        "",
        'echo "Starting simulation groups..."',
        "",
    ]
    for script in group_scripts:
        lines += [
            f'echo "--- Running {script} ---"',
            f'bash "$SCRIPT_DIR/{script}"',
            "",
        ]
    lines += [
        'echo "Waiting for standalone table jobs..."',
        'wait $TABLE4_PID && echo "Table 4 done." '
        '|| echo "Table 4 FAILED -- check logs/table4_bg.log"',
        'wait $TABLE5_PID && echo "Table 5 done." '
        '|| echo "Table 5 FAILED -- check logs/table5_bg.log"',
        "",
        'bash "$SCRIPT_DIR/generate_results_postrun.sh"',
        "",
        "echo '==================================================================='",
        "echo 'All done! Results are in Sat_Simulator/results/'",
        "echo \"Time: $(date)\"",
        "",
    ]
    return "\n".join(lines)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--combined-only", action="store_true",
        help="Generate scripts for the 'combined' scenario only (6 individual scripts "
             "instead of 18, 2 group scripts instead of 6). Sufficient to validate "
             "the core claims. Reduces total wall-clock time from ~9 days to ~3 days.",
    )
    args = parser.parse_args()

    here        = os.path.dirname(os.path.abspath(__file__))
    script_dir  = os.path.join(here, "batch_scripts")
    indiv_dir   = os.path.join(script_dir, "individual")
    logs_dir    = os.path.join(here, "logs")
    for d in (script_dir, indiv_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    scenarios      = ["combined"] if args.combined_only else ALL_SCENARIOS
    group_files: list = []

    # ------------------------------------------------------------------
    # 1. Individual scripts  (one per simulation run)
    # ------------------------------------------------------------------
    print(f"Generating individual scripts  --> batch_scripts/individual/")
    for scenario, (mode_label, mode_arg), hw in product(scenarios, MODES, HARDWARE):
        filename = f"run_{scenario}_{mode_label}_{hw}.sh"
        path     = os.path.join(indiv_dir, filename)
        _write(path, _individual_script(scenario, hw, mode_label, mode_arg))
        print(f"  individual/{filename}  (~{WALL_CLOCK_H} h)")

    # ------------------------------------------------------------------
    # 2. Group scripts  (one per scenario x hardware, all 3 modes)
    # ------------------------------------------------------------------
    print(f"\nGenerating group scripts  --> batch_scripts/")
    for scenario, hw in product(scenarios, HARDWARE):
        filename = f"run_{scenario}_{hw}.sh"
        path     = os.path.join(script_dir, filename)
        _write(path, _group_script(scenario, hw, MODES))
        group_files.append(filename)
        print(f"  {filename}  (~{len(MODES) * WALL_CLOCK_H} h)")

    # ------------------------------------------------------------------
    # 3. Combined-scenario convenience scripts
    # ------------------------------------------------------------------
    print(f"\nGenerating combined-scenario convenience scripts...")
    both_path = os.path.join(script_dir, "run_combined_both.sh")
    _write(both_path, _combined_both_script())
    print(f"  run_combined_both.sh  (~{2 * len(MODES) * WALL_CLOCK_H} h)")

    # ------------------------------------------------------------------
    # 4. Standalone table scripts
    # ------------------------------------------------------------------
    print(f"\nGenerating standalone table scripts...")
    for py_script, sh_name, desc, time_est in [
        ("generate_table_4.py", "generate_table_4.sh",
         "Standalone: Table 4 -- compute time benchmark", "~20 min"),
        ("generate_table_5.py", "generate_table_5.sh",
         "Standalone: Table 5 -- multitask oracle benchmark", "~90 min"),
    ]:
        _write(os.path.join(script_dir, sh_name), _table_script(py_script, desc, time_est))
        print(f"  {sh_name}  ({time_est}, no simulation data needed)")

    # ------------------------------------------------------------------
    # 5. Post-simulation results script
    # ------------------------------------------------------------------
    print(f"\nGenerating post-simulation script...")
    _write(os.path.join(script_dir, "generate_results_postrun.sh"), _postrun_script())
    print(f"  generate_results_postrun.sh  (seconds; run after all sims finish)")

    # ------------------------------------------------------------------
    # 6. All-in-one helpers
    # ------------------------------------------------------------------
    print(f"\nGenerating all-in-one helpers...")
    combined_group_files = [f for f in group_files if "combined" in f]

    combined_all_path = os.path.join(script_dir, "run_all_combined.sh")
    _write(combined_all_path, _run_all_script(combined_group_files))
    combined_total_h = len(combined_group_files) * len(MODES) * WALL_CLOCK_H
    print(f"  run_all_combined.sh  (combined only, ~{combined_total_h} h / ~{combined_total_h // 24} days)")

    if not args.combined_only:
        full_all_path = os.path.join(script_dir, "run_all_full.sh")
        _write(full_all_path, _run_all_script(group_files))
        full_total_h = len(group_files) * len(MODES) * WALL_CLOCK_H
        print(f"  run_all_full.sh  (all scenarios, ~{full_total_h} h / ~{full_total_h // 24} days)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_indiv = len(scenarios) * len(MODES) * len(HARDWARE)
    scope   = "combined scenario only" if args.combined_only else "all scenarios"
    print(f"\n{'='*64}")
    print(f"Scope          : {scope}")
    print(f"Individual scrp: {n_indiv}  (in batch_scripts/individual/)")
    print(f"Group scripts  : {len(group_files)}  (in batch_scripts/)")
    print(f"Wall-clock/run : ~{WALL_CLOCK_H} h  ({SIMULATION_HOURS} simulated hours)")
    print(f"RAM required   : 256 GB")
    print(f"Output dir     : {script_dir}")
    print()
    print("Suggested workflow:")
    print("  # Standalone tables -- start immediately, no simulations needed:")
    print("  bash batch_scripts/generate_table_4.sh   # ~20 min")
    print("  bash batch_scripts/generate_table_5.sh   # ~90 min")
    print()
    print("  # Simulations -- choose one approach:")
    print("  bash batch_scripts/run_all_combined.sh   # combined only (~3 days)")
    if not args.combined_only:
        print("  bash batch_scripts/run_all_full.sh       # all scenarios (~9 days)")
    print()
    print("  # Or run a specific combined subset:")
    print("  bash batch_scripts/run_combined_tpu.sh   # combined + TPU (~36 h)")
    print("  bash batch_scripts/run_combined_gpu.sh   # combined + GPU (~36 h)")
    print("  bash batch_scripts/run_combined_both.sh  # both combined groups (~72 h)")
    print()
    print("  # After all simulations finish:")
    print("  bash batch_scripts/generate_results_postrun.sh")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
