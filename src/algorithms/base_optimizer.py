from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Tuple, List, Optional


class BaseOptimizer(ABC):
    """
    Abstract base class for all evolutionary and swarm algorithms.
    Enforces a unified API so the experiment runner and UI can swap
    algorithms (GA, PSO, ACO) seamlessly.
    """

    def __init__(
            self,
            num_features: int,
            max_iterations: int,
            fitness_func: Callable,
            pop_size: int,
            seed: Optional[int] = None
    ):
        self.num_features = num_features
        self.max_iterations = max_iterations
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.seed = seed

        # Isolated Random Number Generator for thread-safe reproducibility
        self.rng = np.random.default_rng(seed)

        # State tracking — reset between independent runs
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('-inf')
        self.fitness_history: List[float] = []

    def update_best(self, solution: np.ndarray, fitness: float) -> None:
        """
        Safely updates the best known solution.
        Must be called inside the evolve() loop whenever a better fitness is found.
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()  # .copy() prevents reference overwrites

    def reset(self, new_seed: Optional[int] = None) -> None:
        """
        Wipes the state clean for a fresh run.
        Accepts a new seed to prevent the "identical runs" trap in the experiment loop.
        """
        self.seed = new_seed if new_seed is not None else self.seed
        self.rng = np.random.default_rng(self.seed)

        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []

    @abstractmethod
    def initialize_population(self) -> np.ndarray:
        """
        Generates the initial population (GA), swarm (PSO), or pheromones (ACO).
        Must return the initialized structure.
        """
        pass

    @abstractmethod
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """
        Calculates fitness for the entire population/swarm.
        Returns a 1D numpy array of fitness scores corresponding to the population.
        """
        pass

    @abstractmethod
    def evolve(self) -> None:
        """
        Executes exactly ONE generation/iteration of the algorithm.
        Includes selection, crossover, mutation, or velocity/position updates.
        """
        pass

    @abstractmethod
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        The main execution loop for a SINGLE run.
        Should loop `max_iterations` times, calling `evolve()` and appending
        the generation's best fitness to `self.fitness_history`.

        Returns:
            Tuple containing:
            - The best feature subset mask (1D numpy array)
            - The best fitness score (float)
            - The convergence history (List of floats)
        """
        pass