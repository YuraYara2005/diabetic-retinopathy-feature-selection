import time
import numpy as np
from pathlib import Path

# Import our custom modules
from src.utils.config_loader import load_config
from src.utils.seeding import generate_experiment_seeds
from src.utils.logger import save_run_results
from src.evaluation.models import get_model
from src.evaluation.fitness import FitnessEvaluator


class DummyOptimizer:
    """A simple mock optimizer to test the pipeline."""

    def __init__(self, num_features, max_iterations, fitness_func, pop_size, seed):
        self.num_features = num_features
        self.max_iterations = max_iterations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self):
        # Generate a random binary array for the feature subset
        best_subset = self.rng.integers(0, 2, size=self.num_features).astype(np.int8)
        # Generate a random fitness score
        best_fitness = float(self.rng.uniform(0.5, 0.99))

        # Simulate convergence history based on user-defined iterations
        history = [best_fitness - (0.01 * i) for i in range(self.max_iterations)]
        history.sort()
        return best_subset, best_fitness, history


# CRITICAL: This line now accepts classifier_override, pop_override, and gen_override
def run_experiments(algo_name="dummy", config_path="experiment_config.yaml", classifier_override=None,
                    pop_override=None, gen_override=None):
    """Runs the algorithm 30 times with UI slider overrides."""
    print(f"Starting experiments for: {algo_name.upper()}")

    # 1. Load config
    config = load_config(config_path)
    exp_cfg = config['experiment']
    data_cfg = config['dataset']

    # Determine which classifier to use (UI override)
    model_name = classifier_override if classifier_override else exp_cfg['classifier']

    # 2. Load dataset safely
    base_dir = Path(__file__).resolve().parent.parent.parent
    X_train = np.load(base_dir / data_cfg['x_train_path'])
    y_train = np.load(base_dir / data_cfg['y_train_path'])
    X_val = np.load(base_dir / data_cfg['x_val_path'])
    y_val = np.load(base_dir / data_cfg['y_val_path'])
    num_features = X_train.shape[1]

    # 3. Setup evaluator
    model = get_model(model_name)
    evaluator = FitnessEvaluator(
        model=model, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val, alpha=exp_cfg['alpha']
    )

    # 4. Seeds
    seeds = generate_experiment_seeds(exp_cfg['master_seed'], exp_cfg['num_runs'])

    # 5. Select Algorithm
    if algo_name.lower() == "dummy":
        OptimizerClass = DummyOptimizer
    else:
        # This will be where you uncomment GA, PSO, etc. once ready
        raise ValueError(f"Algorithm {algo_name} not found!")

    # 6. Run the 30 loops
    start_time = time.time()
    for i, seed in enumerate(seeds):
        run_id = i + 1
        evaluator.cache.clear()

        # UPDATED: Now uses the values from your UI sliders!
        optimizer = OptimizerClass(
            num_features=num_features,
            max_iterations=gen_override if gen_override else exp_cfg.get('max_iterations', 100),
            fitness_func=evaluator.compute_fitness,
            pop_size=pop_override if pop_override else exp_cfg.get('pop_size', 30),
            seed=seed
        )

        best_subset, best_fitness, history = optimizer.run()
        save_run_results(algo_name, run_id, seed, best_fitness, best_subset, history, config)

    print(f"\nAll runs finished in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    run_experiments(algo_name="dummy")