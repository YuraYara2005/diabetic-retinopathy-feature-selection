import numpy as np
from typing import Tuple, List, Optional
from src.algorithms.base_optimizer import BaseOptimizer
from joblib import Parallel, delayed

class GeneticAlgorithm(BaseOptimizer):
    def __init__(
        self,
        num_features: int,
        max_iterations: int,
        fitness_func,
        config: dict,
        seed: Optional[int] = None,
        bestMutationConfig=None
    ):
        """
        Initialize Genetic Algorithm with configurable selection and crossover strategies,
        along with adaptive mutation and diversity control.
        """
        pop_size = config.get("pop_size", 50)
        super().__init__(num_features, max_iterations, fitness_func, pop_size, seed)

        # Selection & crossover strategy
        self.selection_type = config.get("selection_type", "tournament")
        self.crossover_type = config.get("crossover_type", "uniform")

        # Mutation configuration
        self.bestMutationConfig = bestMutationConfig

        rate = None
        if self.bestMutationConfig is not None:
            rate = self.bestMutationConfig.getMutationRate()

        if rate is None:
            rate = config.get("mutation_rate")

        if rate is None:
            rate = 0.01

        self.base_mutation_rate = rate
        self.current_mutation_rate = self.base_mutation_rate

        # GA parameters
        self.crossover_rate = config.get("crossover_rate", 0.85)
        self.init_prob = config.get("init_prob", 0.05)
        self.k_tournament = config.get("k_tournament", 5)

        # Diversity control parameters
        self.patience = config.get("patience", 35)
        self.div_low = config.get("diversity_low", 0.1)
        self.div_high = config.get("diversity_high", 0.35)
        self.div_extreme = config.get("diversity_extreme", 0.02)

        # Mutation bounds
        self.max_mutation_rate = config.get("max_mutation_rate", 0.25)
        self.min_mutation_rate = config.get("min_mutation_rate", 0.005)

        # Random injection to increase diversity
        self.inject_ratio = config.get("inject_ratio", 0.1)

    # =========================
    # Population Initialization
    # =========================
    def initialize_population(self) -> np.ndarray:
        """Generate initial population as binary vectors."""
        return (self.rng.random((self.pop_size, self.num_features)) < self.init_prob).astype(int)

    # =========================
    # Fitness Evaluation
    # =========================
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array(
            Parallel(n_jobs=-1)(
                delayed(self.fitness_func)(ind) for ind in population
            )
        )
    # =========================
    # Diversity Calculation
    # =========================
    def compute_diversity(self, population: np.ndarray) -> float:
        """Compute average Hamming distance between individuals."""
        total_distance = 0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_distance += np.sum(population[i] != population[j])
                count += 1

        return total_distance / (count * self.num_features) if count > 0 else 0.0

    # =========================
    # Selection Methods
    # =========================
    def tournament_selection(self, fitness: np.ndarray, diversity: float) -> int:
        """Select parent using tournament selection (adaptive size based on diversity)."""
        if diversity < self.div_low:
            k = max(2, int(self.k_tournament * diversity * 2))
        else:
            k = self.k_tournament

        idx = self.rng.choice(len(fitness), size=k, replace=False)
        return idx[np.argmax(fitness[idx])]

    def roulette_selection(self, fitness: np.ndarray) -> int:
        """Select parent using roulette wheel (fitness-proportional selection)."""
        probs = (fitness + 1e-8) / np.sum(fitness + 1e-8)
        return self.rng.choice(len(fitness), p=probs)

    def select_parent(self, fitness, diversity):
        """Choose selection strategy based on configuration."""
        if self.selection_type == "tournament":
            return self.tournament_selection(fitness, diversity)
        elif self.selection_type == "roulette":
            return self.roulette_selection(fitness)
        else:
            raise ValueError("Unknown selection type")

    # =========================
    # Crossover Methods
    # =========================
    def uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Uniform crossover: each gene selected randomly from either parent."""
        mask = self.rng.random(self.num_features) < 0.5
        return np.where(mask, p1, p2)

    def one_point_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """One-point crossover: split parents at random point."""
        point = self.rng.integers(1, self.num_features - 1)
        return np.concatenate([p1[:point], p2[point:]])

    def apply_crossover(self, p1, p2):
        """Apply selected crossover strategy with given probability."""
        if self.rng.random() >= self.crossover_rate:
            return p1.copy()

        if self.crossover_type == "uniform":
            return self.uniform_crossover(p1, p2)
        elif self.crossover_type == "one_point":
            return self.one_point_crossover(p1, p2)
        else:
            raise ValueError("Unknown crossover type")

    # =========================
    # Mutation
    # =========================
    def adaptive_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Adaptive mutation based on sparsity of selected features."""
        active = np.sum(individual)
        ratio = active / self.num_features

        if ratio > 0.2:
            flip_prob = self.current_mutation_rate * 2.5
            mask = (self.rng.random(self.num_features) < flip_prob) & (individual == 1)
        else:
            flip_prob = self.current_mutation_rate
            mask = self.rng.random(self.num_features) < flip_prob

        individual[mask] = 1 - individual[mask]
        return individual

    # =========================
    # Evolution Step
    # =========================
    def evolve(self, population, fitness, diversity):
        """Generate next generation using selection, crossover, mutation, and elitism."""
        new_population = []

        elite_idx = np.argmax(fitness)
        new_population.append(population[elite_idx].copy())

        if diversity < self.div_extreme:
            self.current_mutation_rate = self.max_mutation_rate
        elif diversity < self.div_low:
            scale = (self.div_low - diversity) / self.div_low
            self.current_mutation_rate = self.base_mutation_rate + scale * (
                self.max_mutation_rate - self.base_mutation_rate
            )
        elif diversity > self.div_high:
            self.current_mutation_rate = max(
                self.min_mutation_rate,
                self.base_mutation_rate * 0.5
            )
        else:
            self.current_mutation_rate = self.base_mutation_rate

        inject_count = int(self.inject_ratio * self.pop_size) if diversity < self.div_extreme else 0

        while len(new_population) < self.pop_size:

            if inject_count > 0:
                new_individual = (self.rng.random(self.num_features) < self.init_prob).astype(int)
                new_population.append(new_individual)
                inject_count -= 1
                continue

            p1 = population[self.select_parent(fitness, diversity)]
            p2 = population[self.select_parent(fitness, diversity)]

            child = self.apply_crossover(p1, p2)
            child = self.adaptive_mutation(child)

            new_population.append(child)

        return np.array(new_population)

    # =========================
    # Main Optimization Loop
    # =========================
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run the genetic algorithm optimization process."""
        population = self.initialize_population()
        no_improvement_counter = 0

        for gen in range(self.max_iterations):
            fitness_scores = self.evaluate(population)
            best_idx = np.argmax(fitness_scores)

            if fitness_scores[best_idx] > self.best_fitness:
                self.update_best(population[best_idx], fitness_scores[best_idx])
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            self.fitness_history.append(self.best_fitness)

            diversity = self.compute_diversity(population)

            print(
                f"Gen {gen + 1:02d} | "
                f"Best: {self.best_fitness:.4f} | "
                f"Div: {diversity:.4f} | "
                f"Mut: {self.current_mutation_rate:.4f}"
            )

            if no_improvement_counter >= 20:
                print("--> Stop: no improvement for 20 generations")
                break

            population = self.evolve(population, fitness_scores, diversity)

        return self.best_solution, self.best_fitness, self.fitness_history