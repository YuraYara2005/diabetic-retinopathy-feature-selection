import time
import numpy as np
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
        self.num_features    = num_features
        self.max_iterations  = max_iterations
        self.pop_size        = pop_size
        self.seed            = seed
        self.rng             = np.random.default_rng(seed)

    def run(self):
        peak     = float(self.rng.uniform(0.72, 0.95))
        start    = peak - self.rng.uniform(0.10, 0.25)
        history  = list(np.linspace(start, peak, self.max_iterations))
        history  = [h + float(self.rng.uniform(-0.005, 0.005)) for h in history]

        best_subset = self.rng.integers(0, 2, size=self.num_features).astype(np.int8)
        if best_subset.sum() == 0:
            best_subset[0] = 1

        return best_subset, peak, history


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run_experiments(
    algo_name: str           = "dummy",
    config_path: str         = "experiment_config.yaml",
    classifier_override: Optional[str]  = None,
    pop_override:        Optional[int]  = None,
    gen_override:        Optional[int]  = None,
    progress_callback:   Optional[Callable[[float], None]] = None,
):
    """
    Runs the selected algorithm for num_runs independent experiments.

    Args:
        algo_name:           "dummy" | "pso" | "ga"
        config_path:         Path to experiment_config.yaml
        classifier_override: UI classifier choice overrides the yaml value
        pop_override:        UI population slider overrides the yaml value
        gen_override:        UI generations slider overrides the yaml value
        progress_callback:   Optional callable(float 0→1) for live progress bar
    """
    algo_name = algo_name.lower().strip()
    print(f"\n{'='*50}")
    print(f"  Starting: {algo_name.upper()}")
    print(f"{'='*50}")

    # ── 1. Load config ──────────────────────────────
    config  = load_config(config_path)
    exp_cfg = config["experiment"]
    data_cfg = config["dataset"]

    num_runs       = exp_cfg["num_runs"]
    master_seed    = exp_cfg["master_seed"]
    alpha          = exp_cfg["alpha"]
    model_name     = classifier_override or exp_cfg["classifier"]
    max_iterations = gen_override  or exp_cfg.get("max_iterations", 100)
    pop_size       = pop_override  or exp_cfg.get("pop_size", 30)

    # ── 2. Select optimizer class ───────────────────
    if algo_name == "dummy":
        OptimizerClass  = DummyOptimizer
        needs_real_data = False

    elif algo_name == "pso":
        from src.algorithms.pso import BinaryPSO
        OptimizerClass  = BinaryPSO
        needs_real_data = True

    elif algo_name == "ga":
        from src.algorithms.ga import GeneticAlgorithm
        OptimizerClass  = GeneticAlgorithm
        needs_real_data = True

    else:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Choose from: dummy, pso, ga"
        )

    # ── 3. Load dataset (skip for dummy) ────────────
    if needs_real_data:
        base_dir = Path(__file__).resolve().parent.parent.parent
        try:
            X_train = np.load(base_dir / data_cfg["x_train_path"])
            y_train = np.load(base_dir / data_cfg["y_train_path"])
            X_val   = np.load(base_dir / data_cfg["x_val_path"])
            y_val   = np.load(base_dir / data_cfg["y_val_path"])
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset file not found: {e.filename}\n"
                f"Run the feature extraction pipeline first, or use algo_name='dummy' for testing."
            ) from e

        num_features = X_train.shape[1]
        model        = get_model(model_name)
        evaluator    = FitnessEvaluator(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            alpha=alpha,
        )
        fitness_func = evaluator.compute_fitness

    else:
        # Dummy uses a fake feature space
        num_features = exp_cfg.get("num_features", 512)
        fitness_func = None   # DummyOptimizer ignores this

    # ── 4. Generate seeds ────────────────────────────
    seeds = generate_experiment_seeds(master_seed, num_runs)

    # ── 5. Run loop ──────────────────────────────────
    start_time = time.time()

    for i, seed in enumerate(seeds):
        run_id = i + 1
        print(f"  Run {run_id:02d}/{num_runs}  seed={seed}", end="  ")

        # Clear cache between runs so fitness evaluations are independent
        if needs_real_data:
            evaluator.cache.clear()

        optimizer = OptimizerClass(
            num_features   = num_features,
            max_iterations = max_iterations,
            fitness_func   = fitness_func,
            pop_size       = pop_size,
            seed           = seed,
        )

        best_subset, best_fitness, history = optimizer.run()

        print(f"fitness={best_fitness:.4f}  features={int(best_subset.sum())}")

        save_run_results(
            algo_name, run_id, seed,
            best_fitness, best_subset, history, config
        )

        # Live progress bar update for Streamlit
        if progress_callback is not None:
            progress_callback((i + 1) / num_runs)

    elapsed = time.time() - start_time
    print(f"\n  ✓ {num_runs} runs finished in {elapsed:.1f}s")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature-selection experiments.")
    parser.add_argument("--algo",       default="dummy",  help="dummy | pso | ga")
    parser.add_argument("--classifier", default=None,     help="knn | svm | random_forest")
    parser.add_argument("--pop",        type=int, default=None, help="Population size")
    parser.add_argument("--gen",        type=int, default=None, help="Generations / iterations")
    parser.add_argument("--config",     default="experiment_config.yaml")
    args = parser.parse_args()

    run_experiments(
        algo_name           = args.algo,
        config_path         = args.config,
        classifier_override = args.classifier,
        pop_override        = args.pop,
        gen_override        = args.gen,
    )