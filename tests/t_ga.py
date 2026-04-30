from pathlib import Path
import numpy as np

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
# Config
# ===============================
config = {
    "pop_size": 100,
    "num_generations": 100,
    "mutation_rate": 0.015,
    "crossover_rate": 0.85,
    "init_prob": 0.02,
    "k_tournament": 3,
    "patience": 40,
    "diversity_low": 0.1,
    "diversity_extreme": 0.02,
    "inject_ratio": 0.15
}

# ===============================
# Run
# ===============================
model = get_model("RF")

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

best_sol, best_fit, history = ga.run()

# ===============================
# Final evaluation
# ===============================
selected_idx = np.where(best_sol == 1)[0]

X_train_sel = X_train[:, selected_idx]
X_val_sel   = X_val[:, selected_idx]

accuracy = model.train_and_evaluate(
    X_train_sel, y_train,
    X_val_sel, y_val
)

print("\n===== FINAL RESULT =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Selected Features: {len(selected_idx)}")
print(f"Fitness: {best_fit:.4f}")