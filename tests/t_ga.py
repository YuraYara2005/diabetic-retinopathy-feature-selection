from pathlib import Path
import numpy as np
import time

from src.algorithms.ga import GeneticAlgorithm
from src.evaluation.fitness import FitnessEvaluator
from src.evaluation.models import get_model

# ===============================
# Load data
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent

X_train = np.load(BASE_DIR / "data/processed/train_features.npy")
y_train = np.load(BASE_DIR / "data/processed/train_labels.npy")

X_val   = np.load(BASE_DIR / "data/processed/val_features.npy")
y_val   = np.load(BASE_DIR / "data/processed/val_labels.npy")

# ===============================
# Base Config (optimized)
# ===============================
base_config = {
    "pop_size": 40,
    "num_generations": 60,
    "mutation_rate": 0.015,
    "crossover_rate": 0.85,
    "init_prob": 0.01,
    "k_tournament": 5,
    "patience": 20,
    "diversity_low": 0.1,
    "diversity_extreme": 0.02,
    "inject_ratio": 0.1
}

# ===============================
# Experiments
# ===============================
experiments = [
    {"name": "Tournament + Uniform", "selection": "tournament", "crossover": "uniform"},
    {"name": "Tournament + OnePoint", "selection": "tournament", "crossover": "one_point"},
    {"name": "Roulette + Uniform", "selection": "roulette", "crossover": "uniform"},
    {"name": "Roulette + OnePoint", "selection": "roulette", "crossover": "one_point"},
]

results = []

# ===============================
# Run experiments
# ===============================
for exp in experiments:

    print(f"\nRunning: {exp['name']}")

    config = base_config.copy()
    config["selection_type"] = exp["selection"]
    config["crossover_type"] = exp["crossover"]

    model = get_model("rf")

    fitness = FitnessEvaluator(
        model,
        X_train, y_train,
        X_val, y_val,
        alpha=0.92,
        penalty=0.08
    )

    ga = GeneticAlgorithm(
        num_features=X_train.shape[1],
        max_iterations=config["num_generations"],
        fitness_func=fitness.compute_fitness,
        config=config,
        seed=42
    )

    start = time.time()

    best_sol, best_fit, history = ga.run()

    end = time.time()

    selected_idx = np.where(best_sol == 1)[0]

    X_train_sel = X_train[:, selected_idx]
    X_val_sel   = X_val[:, selected_idx]

    accuracy = model.train_and_evaluate(
        X_train_sel, y_train,
        X_val_sel, y_val
    )

    result = {
        "name": exp["name"],
        "accuracy": accuracy,
        "features": len(selected_idx),
        "fitness": best_fit,
        "time": end - start
    }

    results.append(result)

# ===============================
# Print Results
# ===============================
print("\n===== COMPARISON RESULTS =====")

for r in results:
    print(f"\n{r['name']}")
    print(f"Accuracy: {r['accuracy']:.4f}")
    print(f"Features: {r['features']}")
    print(f"Fitness: {r['fitness']:.4f}")
    print(f"Time: {r['time']:.2f} sec")