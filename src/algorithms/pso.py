import numpy as np
from src.algorithms.base_optimizer import BaseOptimizer


class BinaryPSO(BaseOptimizer):

    def __init__(
        self,
        num_features,
        max_iterations,
        fitness_func,
        pop_size,
        c1=1.5,
        c2=1.5,
        w=0.7,
        seed=None
    ):
        super().__init__(
            num_features,
            max_iterations,
            fitness_func,
            pop_size,
            seed
        )

        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.velocities = None
        self.population = None

        self.pbest_positions = None
        self.pbest_scores = None

        self.gbest_position = None


    def initialize_population(self):

        self.population = self.rng.integers(
            0, 2,
            size=(
                self.pop_size,
                self.num_features
            )
        )

        for particle in self.population:
            if particle.sum() == 0:
                idx = self.rng.integers(
                    self.num_features
                )
                particle[idx] = 1


        self.velocities = self.rng.uniform(
            -4, 4,
            size=(
                self.pop_size,
                self.num_features
            )
        )

        self.pbest_positions = \
            self.population.copy()

        self.pbest_scores = \
            self.evaluate(
                self.population
            )

        best_idx = np.argmax(
            self.pbest_scores
        )

        self.gbest_position = \
            self.population[
                best_idx
            ].copy()

        self.update_best(
            self.gbest_position,
            self.pbest_scores[best_idx]
        )

        return self.population
    
    
    def evaluate(self, population):

        fitness_scores = []

        for particle in population:

            score = self.fitness_func(
                particle
            )

            fitness_scores.append(
                score
            )

        return np.array(
            fitness_scores
        )
    

    def sigmoid(self, x):
        return 1 / (
            1 + np.exp(-x)
        )
    

    def evolve(self):

        for i in range(
            self.pop_size
        ):

            r1 = self.rng.random(
                self.num_features
            )

            r2 = self.rng.random(
                self.num_features
            )


            cognitive = (
                self.c1 *
                r1 *
                (
                self.pbest_positions[i]
                - self.population[i]
                )
            )

            social = (
                self.c2 *
                r2 *
                (
                self.gbest_position
                - self.population[i]
                )
            )


            self.velocities[i] = (
                self.w *
                self.velocities[i]
                + cognitive
                + social
            )

            # to prevent sigmoid saturation
            self.velocities[i] = np.clip(
                self.velocities[i],
                -4,
                4
            )


            probs = self.sigmoid(
                self.velocities[i]
            )


            random_vals = self.rng.random(
                self.num_features
            )

            self.population[i] = (
                random_vals < probs
            ).astype(int)


            # avoid empty feature subset
            if self.population[i].sum()==0:
                idx=self.rng.integers(
                    self.num_features
                )
                self.population[i][idx]=1


            fitness = self.fitness_func(
                self.population[i]
            )


            if fitness > self.pbest_scores[i]:
                self.pbest_scores[i] = fitness

                self.pbest_positions[i] = \
                    self.population[i].copy()


            if fitness > self.best_fitness:
                self.gbest_position = \
                    self.population[i].copy()

                self.update_best(
                    self.population[i],
                    fitness
                )


        self.fitness_history.append(
            float(self.best_fitness)
        )


    def run(self):

        self.initialize_population()

        for _ in range(
            self.max_iterations
        ):
            self.evolve()

        return (
            self.best_solution,
            self.best_fitness,
            self.fitness_history
        )