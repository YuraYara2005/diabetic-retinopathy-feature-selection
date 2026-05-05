"""
analyze_results.py
==================
Post-hoc statistical analysis of Evolutionary Algorithm (EA) feature-selection
experiments on the Diabetic Retinopathy dataset.

We run each EA 30 independent times to ensure our results are statistically
meaningful (i.e., not a lucky single run). This script aggregates those 30
result files, computes descriptive statistics, and visualises the average
convergence behaviour with its variance band.

Usage:
    python analyze_results.py
    # To switch algorithm, change the ALGO_NAME constant at the bottom.

Dependencies:
    numpy, matplotlib
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# =============================================================================
# SECTION 1 – Data Loading
# =============================================================================

def load_run_results(results_dir: Path) -> list[dict]:
    """
    Glob all run_XX.json files from a results directory and return their
    parsed contents as a list of dictionaries.

    Each dictionary is guaranteed to have these keys (as produced by
    save_run_results in the experiment pipeline):
        - run_id             (int)
        - seed_used          (int)
        - best_fitness       (float)
        - features_selected  (int)
        - convergence_history (list[float])

    Parameters
    ----------
    results_dir : Path
        Path to the directory containing the JSON result files, e.g.
        ``results/GA/``.

    Returns
    -------
    list[dict]
        List of parsed run dictionaries, sorted by run_id.

    Raises
    ------
    FileNotFoundError
        If the results directory does not exist.
    ValueError
        If no JSON files are found inside the directory.
    """
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: '{results_dir}'\n"
            "Have you run the experiments yet?"
        )

    # Glob all JSON files and sort them so run_01 comes before run_02, etc.
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        raise ValueError(
            f"No JSON result files found in '{results_dir}'.\n"
            "Make sure the experiment pipeline saved its outputs there."
        )

    runs = []
    for filepath in json_files:
        with open(filepath, "r") as f:
            runs.append(json.load(f))

    # Sort by run_id as a safety measure (glob order can vary by OS)
    runs.sort(key=lambda r: r["run_id"])

    print(f"  Loaded {len(runs)} result files from '{results_dir}'")
    return runs


# =============================================================================
# SECTION 2 – Data Aggregation
# =============================================================================

def analyze_algorithm_runs(algo_name: str) -> dict:
    """
    Load and aggregate the 30 independent EA runs for a given algorithm.
    """
    # Build the path dynamically
    current_dir = Path(__file__).resolve().parent

    # If the script is ALREADY inside the 'results' folder,
    # we don't want to look for 'results/results/dummy'
    if current_dir.name == "results":
        results_dir = current_dir / algo_name
    else:
        results_dir = current_dir / "results" / algo_name

    # Check if the folder exists before trying to load
    if not results_dir.exists():
        # Let's try one more place: check the project root
        project_root = current_dir.parent
        results_dir = project_root / "results" / algo_name

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found at: '{results_dir}'\n"
            "Make sure your experiment_runner.py has finished saving the JSON files."
        )

    runs = load_run_results(results_dir)

    # --- Extract scalars ---
    fitness_scores = np.array([r["best_fitness"] for r in runs], dtype=float)

    # Extract accuracy scores (defaults to 0.0 if loading an older run file)
    accuracy_scores = np.array([r.get("best_accuracy", 0.0) for r in runs], dtype=float)

    feature_counts = np.array([r["features_selected"] for r in runs], dtype=int)

    # --- Extract and align convergence histories ---
    histories_raw = [r["convergence_history"] for r in runs]
    max_iters = max(len(h) for h in histories_raw)

    padded_histories = []
    for h in histories_raw:
        if len(h) < max_iters:
            pad_value = h[-1] if h else 0.0
            h = h + [pad_value] * (max_iters - len(h))
        padded_histories.append(h)

    histories = np.array(padded_histories, dtype=float)

    # 💥 NEW: Dynamically check for pBest history files (Safe for GA)
    pbest_histories_raw = []
    for r in runs:
        pbest_path = results_dir / f"pbest_run_{r['run_id']}.npy"
        if pbest_path.exists():
            pbest_histories_raw.append(np.load(pbest_path).tolist())

    pbest_histories = None
    if pbest_histories_raw:
        padded_pbest = []
        for h in pbest_histories_raw:
            if len(h) < max_iters:
                pad_value = h[-1] if h else 0.0
                h = h + [pad_value] * (max_iters - len(h))
            padded_pbest.append(h)
        pbest_histories = np.array(padded_pbest, dtype=float)

    return {
        "algo_name": algo_name,
        "runs": runs,
        "fitness": fitness_scores,
        "accuracy": accuracy_scores,
        "n_features": feature_counts,
        "histories": histories,
        "pbest_histories": pbest_histories # Will be None if it's a GA
    }


# =============================================================================
# SECTION 3 – Statistical Summary Report
# =============================================================================

def print_statistical_summary(data: dict) -> None:
    """
    Print a formatted terminal report of key statistics across all runs.
    """
    algo    = data["algo_name"]
    fitness = data["fitness"]
    acc     = data.get("accuracy", np.zeros_like(fitness))
    feats   = data["n_features"]
    n_runs  = len(data["runs"])

    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  Statistical Summary  |  Algorithm: {algo}  |  N = {n_runs} runs")
    print(sep)

    # ── Best Fitness ───────────────────────────────────────────────────────
    print("\n  [ Best Fitness Score ]")
    print(f"    Mean  ± Std  :  {fitness.mean():.6f}  ±  {fitness.std(ddof=1):.6f}")
    print(f"    Min          :  {fitness.min():.6f}")
    print(f"    Max          :  {fitness.max():.6f}")

    # ── Best Accuracy ──────────────────────────────────────────────────────
    print("\n  [ Best Accuracy Score ]")
    print(f"    Mean  ± Std  :  {acc.mean()*100:.2f}%  ±  {acc.std(ddof=1)*100:.2f}%")
    print(f"    Min          :  {acc.min()*100:.2f}%")
    print(f"    Max          :  {acc.max()*100:.2f}%")

    # ── Features Selected ──────────────────────────────────────────────────
    print("\n  [ Features Selected ]")
    print(f"    Mean  ± Std  :  {feats.mean():.2f}  ±  {feats.std(ddof=1):.2f}")
    print(f"    Min          :  {feats.min()}")
    print(f"    Max          :  {feats.max()}")

    print(f"\n{sep}\n")


# =============================================================================
# SECTION 4 – Convergence Plot
# =============================================================================

def plot_convergence(data: dict) -> None:
    """
    Plot the mean convergence curve with a ±1 standard deviation shaded band.
    Includes pBest line if it is a PSO run.
    """
    algo      = data["algo_name"]
    histories = data["histories"]   # shape: (n_runs, n_generations)
    pbest_histories = data.get("pbest_histories")

    # --- Compute per-generation statistics ----------------------------------
    mean_curve = histories.mean(axis=0)      # μ_g across runs at each generation
    std_curve  = histories.std(axis=0, ddof=1)  # σ_g (sample std, Bessel-corrected)
    generations = np.arange(1, len(mean_curve) + 1)  # x-axis: 1, 2, 3, …

    # --- Figure setup -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    # Use a clean, modern style
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "axes.spines.top":   False,   # Remove the top spine (looks cleaner)
        "axes.spines.right": False,   # Remove the right spine
    })

    # --- Plot the individual run curves (very faint, for context) -----------
    for run_history in histories:
        ax.plot(
            generations,
            run_history,
            color="#9ecae1",   # Light blue
            alpha=0.15,        # Very transparent so they don't dominate
            linewidth=0.8,
            zorder=1,          # Draw behind everything else
        )

    # --- Plot the ±1 σ shaded band ------------------------------------------
    ax.fill_between(
        generations,
        mean_curve - std_curve,    # Lower bound: μ - σ
        mean_curve + std_curve,    # Upper bound: μ + σ
        color="#2171b5",
        alpha=0.18,
        label=r"Mean $\pm 1\sigma$ (variance band)",
        zorder=2,
    )

    # --- Plot the mean convergence line -------------------------------------
    ax.plot(
        generations,
        mean_curve,
        color="#08519c",       # Dark blue
        linewidth=2.2,
        label=f"Mean gBest (N = {len(histories)} runs)",
        zorder=3,
    )

    # 💥 NEW: Plot the pBest line if the data exists!
    if pbest_histories is not None and len(pbest_histories) > 0:
        pbest_mean = pbest_histories.mean(axis=0)
        ax.plot(
            generations,
            pbest_mean,
            color="#ff7f0e",
            linestyle="--",
            linewidth=2.2,
            label=f"Mean pBest",
            zorder=3
        )

    # --- Mark the final mean fitness ----------------------------------------
    final_fitness = mean_curve[-1]
    ax.axhline(
        final_fitness,
        color="#08519c",
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
        zorder=2,
    )
    ax.text(
        x=generations[-1] * 0.98,
        y=final_fitness,
        s=f" μ_final = {final_fitness:.4f}",
        va="bottom",
        ha="right",
        fontsize=9,
        color="#08519c",
    )

    # --- Labels, title, legend ----------------------------------------------
    ax.set_xlabel("Generation", fontsize=12, labelpad=8)
    ax.set_ylabel("Fitness Score", fontsize=12, labelpad=8)
    ax.set_title(
        f"{algo} – Convergence Curve over {len(histories)} Independent Runs\n"
        f"Diabetic Retinopathy Feature Selection",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(fontsize=10, loc="lower right", framealpha=0.7)

    # Force integer ticks on the x-axis (generations are whole numbers)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    # --- Save and display ---------------------------------------------------
    script_dir  = Path(__file__).resolve().parent
    output_path = script_dir / "results" / algo / f"{algo}_convergence.png"

    # Ensure the folder exists before saving the plot to prevent crashes
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"  Convergence plot saved → {output_path}")

    plt.show()


# =============================================================================
# SECTION 5 – Main entry point
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Change ALGO_NAME to analyse a different algorithm's results.
    # Valid options (once experiments are run): "GA", "PSO", "ACO", "dummy"
    # -------------------------------------------------------------------------
    ALGO_NAME = "pso_low"

    print(f"\nAnalysing results for: {ALGO_NAME}")
    print("-" * 40)

    # Step 1: Load and aggregate all 30 run files
    data = analyze_algorithm_runs(ALGO_NAME)

    # Step 2: Print the statistical summary table to the terminal
    print_statistical_summary(data)

    # Step 3: Generate and save the convergence plot
    plot_convergence(data)