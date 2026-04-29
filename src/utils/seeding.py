import numpy as np
from typing import List

def generate_experiment_seeds(master_seed: int, n_runs: int = 30) -> List[int]:
    """Generates a deterministic list of sub-seeds for the 30 independent runs."""
    master_rng = np.random.default_rng(master_seed)
    # Generate random integers between 1 and 1,000,000 to use as sub-seeds
    seeds = master_rng.integers(1, 1000000, size=n_runs).tolist()
    return seeds