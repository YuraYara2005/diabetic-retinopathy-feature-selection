import numpy as np
from typing import Dict, Tuple


class FitnessEvaluator:

    def __init__(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        alpha: float = 0.92,
        penalty: float = 0.08,
        min_features: int = 10
    ):
        """
        alpha: importance of accuracy (0.7 → 0.9)
        penalty: punishment for too many features (0.1 → 0.3)
        min_features: prevent trivial solutions (0 features)
        """

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.alpha = alpha
        self.penalty = penalty
        self.min_features = min_features

        self.cache: Dict[Tuple[int], float] = {}

    # ===============================
    # Main Fitness Function
    # ===============================
    def compute_fitness(self, solution: np.ndarray) -> float:

        key = tuple(solution)

        # 🔁 Cache
        if key in self.cache:
            return self.cache[key]

        selected_idx = np.where(solution == 1)[0]
        selected_count = len(selected_idx)

        # no features
        if selected_count == 0:
            self.cache[key] = 0.0
            return 0.0

        # too few features (unstable model)
        if selected_count < self.min_features:
            self.cache[key] = 0.0
            return 0.0

        # ===============================
        # Prepare data
        # ===============================
        X_train_sel = self.X_train[:, selected_idx]
        X_val_sel = self.X_val[:, selected_idx]

        # ===============================
        # Model evaluation
        # ===============================
        try:
            accuracy = self.model.train_and_evaluate(
                X_train_sel, self.y_train,
                X_val_sel, self.y_val
            )
        except Exception:
            self.cache[key] = 0.0
            return 0.0

        # ===============================
        # Feature ratio
        # ===============================
        total_features = self.X_train.shape[1]
        ratio = selected_count / total_features

        # ===============================
        # Final Fitness (BEST FORMULA)
        # ===============================
        fitness = (
                self.alpha * accuracy
                - self.penalty * ratio
        )

        # ===============================
        # Cache
        # ===============================
        self.cache[key] = fitness

        return fitness

# import numpy as np
# from typing import Dict, Tuple
#
#
# class FitnessEvaluator:
#
#     def __init__(self, model, X_train, y_train, X_val, y_val, alpha=0.8):
#         self.model = model
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_val = X_val
#         self.y_val = y_val
#         self.alpha = alpha
#
#         self.cache: Dict[Tuple[int], float] = {}
#
#     def compute_fitness(self, solution: np.ndarray) -> float:
#
#         key = tuple(solution)
#
#         if key in self.cache:
#             return self.cache[key]
#
#         if np.sum(solution) == 0:
#             self.cache[key] = 0.0
#             return 0.0
#
#         selected_idx = np.where(solution == 1)[0]
#
#         X_train_sel = self.X_train[:, selected_idx]
#         X_val_sel = self.X_val[:, selected_idx]
#
#         # 🔥 هنا بقى بنستخدم Task 3
#         accuracy = self.model.train_and_evaluate(
#             X_train_sel, self.y_train,
#             X_val_sel, self.y_val
#         )
#
#         reduction = 1 - (len(selected_idx) / self.X_train.shape[1])
#
#         fitness = self.alpha * accuracy + (1 - self.alpha) * reduction
#
#         self.cache[key] = fitness
#
#         return fitness
