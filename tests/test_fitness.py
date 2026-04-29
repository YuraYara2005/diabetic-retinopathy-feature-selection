import numpy as np
from src.evaluation.fitness  import FitnessEvaluator


def test_fitness_basic():

    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)

    X_val = np.random.rand(5, 5)
    y_val = np.random.randint(0, 2, 5)

    class DummyModel:
        def train_and_evaluate(self, X_train, y_train, X_val, y_val):
            return 1.0

    model = DummyModel()

    evaluator = FitnessEvaluator(
        model,
        X_train, y_train,
        X_val, y_val,
        alpha=0.9
    )

    solution = np.array([1, 1, 1, 1, 1])

    fitness = evaluator.compute_fitness(solution)

    expected = 0.9 * 1 + 0.1 * 0

    print("fitness:", fitness)
    print("expected:", expected)

    assert abs(fitness - expected) < 1e-6


# 👇 تشغيل الدالة هنا
test_fitness_basic()