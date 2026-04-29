import numpy as np
from src.algorithms.pso import BinaryPSO


def dummy_fitness(mask):
    # pretend selecting more useful features
    return np.sum(mask)/len(mask)


pso = BinaryPSO(
    num_features=20,
    max_iterations=10,
    fitness_func=dummy_fitness,
    pop_size=10
)


best_sol,best_fit,history = pso.run()

print("Best fitness:",best_fit)
print("Selected features:",best_sol.sum())
print("History:",history)


""" 
output sample :

Best fitness: 0.85
Selected features: 17
History: [np.float64(0.7), np.float64(0.85), np.float64(0.85), np.float64(0.85), np.float64(0.85), np.float64(0.85), np.float64(0.85), np.float64(0.85), np.float64(0.85), np.float64(0.85)]

"""