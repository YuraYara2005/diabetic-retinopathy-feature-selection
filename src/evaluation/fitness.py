import numpy as np
from typing import Dict, Tuple


class FitnessEvaluator:

    def __init__(self, model, X_train, y_train, X_val, y_val, alpha=0.9):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.alpha = alpha

        self.cache: Dict[Tuple[int], float] = {}

    def compute_fitness(self, solution: np.ndarray) -> float:

        key = tuple(solution)

        if key in self.cache:
            return self.cache[key]

        if np.sum(solution) == 0:
            self.cache[key] = 0.0
            return 0.0

        selected_idx = np.where(solution == 1)[0]

        X_train_sel = self.X_train[:, selected_idx]
        X_val_sel = self.X_val[:, selected_idx]

        # 🔥 هنا بقى بنستخدم Task 3
        accuracy = self.model.train_and_evaluate(
            X_train_sel, self.y_train,
            X_val_sel, self.y_val
        )

        reduction = 1 - (len(selected_idx) / self.X_train.shape[1])

        fitness = self.alpha * accuracy + (1 - self.alpha) * reduction

        self.cache[key] = fitness

        return fitness