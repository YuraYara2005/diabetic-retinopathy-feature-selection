import numpy as np
from src.algorithms.base_optimizer import BaseOptimizer


class BinaryPSO(BaseOptimizer):
    """
    Canonical Global-best Binary PSO for feature selection.

    Key fixes over v1:
    1. Negative velocity initialisation — particles start SPARSE (few features selected).
    2. Linear inertia weight decay (w_max → w_min) so the swarm transitions
       from exploration to exploitation over time (canonical PSO, Shi & Eberhart 1998).
    3. Conservative c1/c2 defaults for high-dimensional binary spaces.
    """

    def __init__(
        self,
        num_features: int,
        max_iterations: int,
        fitness_func,
        pop_size: int,
        c1: float = 0.5,       # personal / cognitive acceleration
        c2: float = 0.3,       # social acceleration
        w_max: float = 0.9,    # inertia at iteration 0  (explore)
        w_min: float = 0.4,    # inertia at last iteration (exploit)
        v_min: float = -6.0,   # velocity clamp lower bound
        v_max: float = 6.0,    # velocity clamp upper bound
        seed=None,
    ):
        super().__init__(num_features, max_iterations, fitness_func, pop_size, seed)

        self.c1    = c1
        self.c2    = c2
        self.w_max = w_max
        self.w_min = w_min
        self.v_min = v_min
        self.v_max = v_max

        # Runtime state — all set in initialize_population()
        self.population      = None
        self.velocities      = None
        self.pbest_positions = None
        self.pbest_scores    = None
        self.gbest_position  = None
        self.current_iter    = 0   # tracks inertia decay

    # ------------------------------------------------------------------ #
    #  INITIALISATION                                                      #
    # ------------------------------------------------------------------ #
    def initialize_population(self) -> np.ndarray:
        """
        Initialises particles with a sparse binary representation.
        Each particle starts with ~10-20% features selected by biasing
        velocities to be negative (sigmoid maps negatives → low probability).
        """
        # --- Binary positions: start ~20% selected on average ---
        # We use a low probability (0.2) so initial subsets are small
        self.population = (
            self.rng.random(size=(self.pop_size, self.num_features)) < 0.2
        ).astype(int)

        # Guarantee no all-zero particle
        for particle in self.population:
            if particle.sum() == 0:
                particle[self.rng.integers(self.num_features)] = 1

        # --- Velocities: biased negative so sigmoid ≈ 0.1-0.27 ---
        # This prevents the "stuck at 50% features" problem in high dimensions
        self.velocities = self.rng.uniform(
            -6, -2,
            size=(self.pop_size, self.num_features)
        )

        # --- Personal bests = starting positions ---
        self.pbest_positions = self.population.copy()
        self.pbest_scores    = self.evaluate(self.population)

        # --- Global best = best particle in initial swarm ---
        best_idx = int(np.argmax(self.pbest_scores))
        self.gbest_position = self.population[best_idx].copy()
        self.update_best(self.gbest_position, self.pbest_scores[best_idx])

        self.current_iter = 0   # reset decay counter

        return self.population

    # ------------------------------------------------------------------ #
    #  EVALUATION                                                          #
    # ------------------------------------------------------------------ #
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Calls the fitness function for every particle in a population array."""
        return np.array([self.fitness_func(p) for p in population])

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def _current_inertia(self) -> float:
        """
        Linear inertia weight decay (Shi & Eberhart, 1998 canonical PSO).
        Starts at w_max (exploration) and decays to w_min (exploitation).
        """
        progress = self.current_iter / max(self.max_iterations - 1, 1)
        return self.w_max - (self.w_max - self.w_min) * progress

    # ------------------------------------------------------------------ #
    #  ONE GENERATION                                                      #
    # ------------------------------------------------------------------ #
    def evolve(self) -> None:
        """
        Performs one complete gbest-PSO iteration over all particles.
        Velocity update: v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        Position update: binarised via sigmoid transfer function.
        """
        w = self._current_inertia()

        for i in range(self.pop_size):
            r1 = self.rng.random(self.num_features)
            r2 = self.rng.random(self.num_features)

            cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
            social    = self.c2 * r2 * (self.gbest_position     - self.population[i])

            # Velocity update (canonical PSO formula with inertia weight)
            self.velocities[i] = w * self.velocities[i] + cognitive + social

            # Clamp to prevent sigmoid saturation
            self.velocities[i] = np.clip(self.velocities[i], self.v_min, self.v_max)

            # Binarise: probability of flipping each bit to 1
            probs = self.sigmoid(self.velocities[i])
            self.population[i] = (self.rng.random(self.num_features) < probs).astype(int)

            # Repair: forbid empty feature subsets
            if self.population[i].sum() == 0:
                self.population[i][self.rng.integers(self.num_features)] = 1

            # Evaluate new position
            fitness = self.fitness_func(self.population[i])

            # Update personal best
            if fitness > self.pbest_scores[i]:
                self.pbest_scores[i]    = fitness
                self.pbest_positions[i] = self.population[i].copy()

            # Update global best
            if fitness > self.best_fitness:
                self.gbest_position = self.population[i].copy()
                self.update_best(self.population[i], fitness)

        # Record best fitness this iteration for convergence plot
        self.fitness_history.append(float(self.best_fitness))
        self.current_iter += 1

    # ------------------------------------------------------------------ #
    #  FULL RUN                                                            #
    # ------------------------------------------------------------------ #
    def run(self):
        """Runs the full PSO for max_iterations and returns results."""
        self.initialize_population()

        for _ in range(self.max_iterations):
            self.evolve()

        return self.best_solution, self.best_fitness, self.fitness_history