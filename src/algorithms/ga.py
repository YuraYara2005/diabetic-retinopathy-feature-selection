import numpy as np
from typing import Tuple, List, Optional
from src.algorithms.base_optimizer import BaseOptimizer
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
        pop_size = config.get("pop_size", 50)
        super().__init__(num_features, max_iterations, fitness_func, pop_size, seed)

        # Mutation rate fallback logic
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

        # Other parameters
        self.crossover_rate = config.get("crossover_rate", 0.85)
        self.init_prob = config.get("init_prob", 0.05)
        self.k_tournament = config.get("k_tournament", 5)

        self.patience = config.get("patience", 35)

        self.div_low = config.get("diversity_low", 0.1)
        self.div_high = config.get("diversity_high", 0.35)
        self.div_extreme = config.get("diversity_extreme", 0.02)

        self.max_mutation_rate = config.get("max_mutation_rate", 0.25)
        self.min_mutation_rate = config.get("min_mutation_rate", 0.005)

        self.inject_ratio = config.get("inject_ratio", 0.1)

    def initialize_population(self) -> np.ndarray:
        return (self.rng.random((self.pop_size, self.num_features)) < self.init_prob).astype(int)

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_func(ind) for ind in population])

    def compute_diversity(self, population: np.ndarray) -> float:
        total_distance = 0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_distance += np.sum(population[i] != population[j])
                count += 1

        return total_distance / (count * self.num_features) if count > 0 else 0.0

    def tournament_selection(self, fitness: np.ndarray, diversity: float) -> int:
        if diversity < self.div_low:
            k = max(2, int(self.k_tournament * diversity * 2))
        else:
            k = self.k_tournament

        idx = self.rng.choice(len(fitness), size=k, replace=False)
        return idx[np.argmax(fitness[idx])]

    def uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.num_features) < 0.5
        return np.where(mask, p1, p2)

    def adaptive_mutation(self, individual: np.ndarray) -> np.ndarray:
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

    def evolve(self, population, fitness, diversity):
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

            p1 = population[self.tournament_selection(fitness, diversity)]
            p2 = population[self.tournament_selection(fitness, diversity)]

            if self.rng.random() < self.crossover_rate:
                child = self.uniform_crossover(p1, p2)
            else:
                child = p1.copy()

            child = self.adaptive_mutation(child)
            new_population.append(child)

        return np.array(new_population)

    def run(self) -> Tuple[np.ndarray, float, List[float]]:
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

