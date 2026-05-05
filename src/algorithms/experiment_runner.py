import time
import os
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Callable

from src.utils.config_loader import load_config
from src.utils.seeding import generate_experiment_seeds
from src.utils.logger import save_run_results
from src.evaluation.models import get_model
from src.evaluation.fitness import FitnessEvaluator


# ─────────────────────────────────────────────
# DUMMY OPTIMIZER  (for pipeline testing only)
# ─────────────────────────────────────────────
class DummyOptimizer:
    """
    Simulates a real optimizer without loading any dataset.
    Used exclusively for UI/pipeline smoke-testing.
    """

    def __init__(self, num_features, max_iterations, fitness_func, pop_size, seed=None):
        self.num_features = num_features
        self.max_iterations = max_iterations
        self.pop_size = pop_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self):
        peak = float(self.rng.uniform(0.72, 0.95))
        start = peak - self.rng.uniform(0.10, 0.25)
        history = list(np.linspace(start, peak, self.max_iterations))
        history = [h + float(self.rng.uniform(-0.005, 0.005)) for h in history]

        best_subset = self.rng.integers(0, 2, size=self.num_features).astype(np.int8)
        if best_subset.sum() == 0:
            best_subset[0] = 1

        return best_subset, peak, history


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run_experiments(
        algo_name: str = "dummy",
        config_path: str = "experiment_config.yaml",
        classifier_override: Optional[str] = None,
        pop_override: Optional[int] = None,
        gen_override: Optional[int] = None,
        runs_override: Optional[int] = None,
        subset_override: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
):
    """
    Runs the selected algorithm for num_runs independent experiments.
    """
    algo_name = algo_name.lower().strip()
    print(f"\n{'=' * 50}")
    print(f"  Starting: {algo_name.upper()}")
    print(f"{'=' * 50}")

    base_dir = Path(__file__).resolve().parent.parent.parent

    # ── 1. Load Master config ──────────────────────────────
    config = load_config(config_path)
    exp_cfg = config["experiment"]
    data_cfg = config["dataset"]

    # Use the UI slider if provided, otherwise fallback to YAML
    num_runs = runs_override or exp_cfg["num_runs"]
    master_seed = exp_cfg["master_seed"]
    alpha = exp_cfg["alpha"]
    model_name = classifier_override or exp_cfg["classifier"]

    # Base parameters (overridden by UI if provided)
    max_iterations = gen_override or exp_cfg.get("max_iterations", 100)
    pop_size = pop_override or exp_cfg.get("pop_size", 30)

    # ── 2. Select optimizer class & Load Specific Configs ──
    algo_config = {}

    if algo_name == "dummy":
        OptimizerClass = DummyOptimizer
        needs_real_data = False


    elif algo_name.startswith("pso"):
        from src.algorithms.pso import BinaryPSO
        OptimizerClass = BinaryPSO
        needs_real_data = True

        # Load PSO config
        try:
            with open(base_dir / "config" / "pso_config.yaml", "r") as f:
                algo_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print("  [!] pso_config.yaml not found. Using defaults.")

        if pop_override is None:
            pop_size = algo_config.get("swarm_size", pop_size)
        if gen_override is None:
            max_iterations = algo_config.get("max_iterations", max_iterations)

        # Inject Inertia parameters based on UI selection
        if "linear" in algo_name:
            algo_config["w_max"] = 0.9
            algo_config["w_min"] = 0.4
        elif "high" in algo_name:
            algo_config["w_max"] = 0.9
            algo_config["w_min"] = 0.9
        elif "low" in algo_name:
            algo_config["w_max"] = 0.4
            algo_config["w_min"] = 0.4

    elif algo_name.startswith("ga"):
        from src.algorithms.ga import GeneticAlgorithm
        OptimizerClass = GeneticAlgorithm
        needs_real_data = True

        # Load GA config
        try:
            with open(base_dir / "config" / "ga_config.yaml", "r") as f:
                algo_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print("  [!] ga_config.yaml not found. Using defaults.")

        # Inject our UI variables directly into the GA's config dictionary
        algo_config["pop_size"] = pop_size
        algo_config["num_generations"] = max_iterations

        # Inject Strategy parameters based on UI selection
        if "tourn" in algo_name:
            algo_config["selection_type"] = "tournament"
        elif "roul" in algo_name:
            algo_config["selection_type"] = "roulette"

        if "uni" in algo_name:
            algo_config["crossover_type"] = "uniform"
        elif "one" in algo_name:
            algo_config["crossover_type"] = "one_point"

    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Choose from: dummy, pso, ga")

    # ── 3. Load dataset (skip for dummy) ────────────
    if needs_real_data:
        try:
            X_train = np.load(base_dir / data_cfg["x_train_path"])
            y_train = np.load(base_dir / data_cfg["y_train_path"])
            X_val = np.load(base_dir / data_cfg["x_val_path"])
            y_val = np.load(base_dir / data_cfg["y_val_path"])


            subset_pct = subset_override if subset_override is not None else 100
            if subset_pct < 100:
                sample_size = int(len(X_train) * (subset_pct / 100.0))
                np.random.seed(42) # Keep it reproducible across algorithms
                idx = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
                print(f"  [⚡] Fast Mode: Using {subset_pct}% of training data ({sample_size} samples).")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset file not found: {e.filename}\n"
                f"Run the feature extraction pipeline first, or use algo_name='dummy' for testing."
            ) from e

        num_features = X_train.shape[1]
        model = get_model(model_name)
        evaluator = FitnessEvaluator(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            alpha=alpha,
        )
        fitness_func = evaluator.compute_fitness

    else:
        num_features = exp_cfg.get("num_features", 512)
        fitness_func = None

    # ── 4. Generate seeds ────────────────────────────
    seeds = generate_experiment_seeds(master_seed, num_runs)

    # ── 5. Run loop ──────────────────────────────────
    start_time = time.time()

    for i, seed in enumerate(seeds):
        run_id = i + 1
        print(f"  Run {run_id:02d}/{num_runs}  seed={seed}", end="  ")

        if needs_real_data:
            evaluator.cache.clear()


        if algo_name.startswith("ga"):
            optimizer = OptimizerClass(
                num_features=num_features,
                max_iterations=max_iterations,
                fitness_func=fitness_func,
                config=algo_config,
                seed=seed,
            )
        elif algo_name.startswith("pso"):
            optimizer = OptimizerClass(
                num_features=num_features,
                max_iterations=max_iterations,
                fitness_func=fitness_func,
                pop_size=pop_size,
                c1=algo_config.get("c1", 0.5),
                c2=algo_config.get("c2", 0.3),
                w_max=algo_config.get("w_max", 0.9),
                w_min=algo_config.get("w_min", 0.4),
                v_min=algo_config.get("velocity_min", -6.0),
                v_max=algo_config.get("velocity_max", 6.0),
                seed=seed,
            )
        else:  # dummy
            optimizer = OptimizerClass(
                num_features=num_features,
                max_iterations=max_iterations,
                fitness_func=fitness_func,
                pop_size=pop_size,
                seed=seed,
            )

        best_subset, best_fitness, history = optimizer.run()


        if needs_real_data and int(best_subset.sum()) > 0:
            selected_idx = np.where(best_subset == 1)[0]
            best_accuracy = evaluator.model.train_and_evaluate(
                X_train[:, selected_idx], y_train,
                X_val[:, selected_idx], y_val
            )
        else:
            best_accuracy = best_fitness if not needs_real_data else 0.0

        print(f"fitness={best_fitness:.4f}  accuracy={best_accuracy:.4f}  features={int(best_subset.sum())}")

        save_run_results(
            algo_name, run_id, seed,
            best_fitness, best_subset, history, config, best_accuracy
        )


        if hasattr(optimizer, "pbest_history"):
            save_dir = base_dir / "results" / algo_name
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / f"pbest_run_{run_id}.npy", optimizer.pbest_history)

        if progress_callback is not None:
            progress_callback((i + 1) / num_runs)

    elapsed = time.time() - start_time
    print(f"\n  ✓ {num_runs} runs finished in {elapsed:.1f}s")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="dummy")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--pop", type=int, default=None)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--config", default="experiment_config.yaml")
    args = parser.parse_args()

    run_experiments(
        algo_name=args.algo,
        config_path=args.config,
        classifier_override=args.classifier,
        pop_override=args.pop,
        gen_override=args.gen,
    )