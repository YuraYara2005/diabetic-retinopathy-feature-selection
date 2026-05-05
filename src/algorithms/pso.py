import numpy as np
from src.algorithms.base_optimizer import BaseOptimizer
from joblib import Parallel, delayed


class BinaryPSO(BaseOptimizer):
    """
    Canonical Global-best Binary PSO for feature selection.
    1. Delta Evaluation (only trains models for particles that change).
    2. Balanced Initialization (starts at 25% features to target the 400-600 range).
    3. Mild Asymmetric Sigmoid (-0.15 penalty to prevent random flipping and boost speed).
    """

    def __init__(
            self,
            num_features: int,
            max_iterations: int,
            fitness_func,
            pop_size: int,
            c1: float = 0.5,
            c2: float = 0.3,
            w_max: float = 0.9,
            w_min: float = 0.4,
            v_min: float = -6.0,
            v_max: float = 6.0,
            seed=None,
    ):
        super().__init__(num_features, max_iterations, fitness_func, pop_size, seed)

        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.v_min = v_min
        self.v_max = v_max

        self.population = None
        self.prev_population = None
        self.current_fitness = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_scores = None
        self.gbest_position = None
        self.current_iter = 0
        # Empty list to quietly store the pBest history for the charts
        self.pbest_history = []

    def initialize_population(self) -> np.ndarray:
        # Start at 25% (roughly 512 features out of 2048)

        self.population = (
                self.rng.random(size=(self.pop_size, self.num_features)) < 0.25
        ).astype(int)

        zero_rows = np.where(self.population.sum(axis=1) == 0)[0]
        for row in zero_rows:
            self.population[row, self.rng.integers(self.num_features)] = 1

        #  Start velocities neutral to slightly negative
        self.velocities = self.rng.uniform(
            -2, 1,
            size=(self.pop_size, self.num_features)
        )

        self.current_fitness = self.evaluate(self.population)
        self.prev_population = self.population.copy()

        self.pbest_positions = self.population.copy()
        self.pbest_scores = self.current_fitness.copy()

        best_idx = int(np.argmax(self.pbest_scores))
        self.gbest_position = self.population[best_idx].copy()
        self.update_best(self.gbest_position, self.pbest_scores[best_idx])

        self.current_iter = 0
        return self.population

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.fitness_func)(ind) for ind in population
            )
        )

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        sig = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

        return np.clip(sig - 0.15, 0.0, 1.0)

    def _current_inertia(self) -> float:
        progress = self.current_iter / max(self.max_iterations - 1, 1)
        return self.w_max - (self.w_max - self.w_min) * progress

    def evolve(self) -> None:
        w = self._current_inertia()

        r1 = self.rng.random((self.pop_size, self.num_features))
        r2 = self.rng.random((self.pop_size, self.num_features))

        cognitive = self.c1 * r1 * (self.pbest_positions - self.population)
        social = self.c2 * r2 * (self.gbest_position - self.population)

        self.velocities = w * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, self.v_min, self.v_max)

        probs = self.sigmoid(self.velocities)
        self.population = (self.rng.random((self.pop_size, self.num_features)) < probs).astype(int)

        zero_rows = np.where(self.population.sum(axis=1) == 0)[0]
        for row in zero_rows:
            self.population[row, self.rng.integers(self.num_features)] = 1

        # Delta Evaluation: Only train models for particles that ACTUALLY changed bits!
        changed_mask = (self.population != self.prev_population).any(axis=1)
        changed_idx = np.where(changed_mask)[0]

        if len(changed_idx) > 0:
            new_fitness = self.evaluate(self.population[changed_idx])
            self.current_fitness[changed_idx] = new_fitness

        self.prev_population = self.population.copy()

        # Update P-Best
        improved_idx = self.current_fitness > self.pbest_scores
        self.pbest_scores[improved_idx] = self.current_fitness[improved_idx]
        self.pbest_positions[improved_idx] = self.population[improved_idx].copy()

        # Update G-Best
        best_current_idx = np.argmax(self.current_fitness)
        if self.current_fitness[best_current_idx] > self.best_fitness:
            self.gbest_position = self.population[best_current_idx].copy()
            self.update_best(self.gbest_position, self.current_fitness[best_current_idx])

        self.fitness_history.append(float(self.best_fitness))
        self.current_iter += 1

    def run(self):
        self.initialize_population()

        for iter_num in range(self.max_iterations):
            self.evolve()

            current_gbest_fitness = self.best_fitness
            current_gbest_feats = int(self.gbest_position.sum())
            avg_pbest_fitness = np.mean(self.pbest_scores)
            current_w = self._current_inertia()

            # Save it to our custom history list for visualizations
            self.pbest_history.append(float(avg_pbest_fitness))

            print(
                f"Iter {iter_num + 1:03d}/{self.max_iterations} | "
                f"gBest Fit: {current_gbest_fitness:.4f} ({current_gbest_feats:04d} feats) | "
                f"Avg pBest: {avg_pbest_fitness:.4f} | "
                f"Inertia (w): {current_w:.4f}"
            )

        return self.best_solution, self.best_fitness, self.fitness_history