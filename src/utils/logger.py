import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

def save_run_results(
        algo_name: str,
        run_id: int,
        seed: int,
        best_fitness: float,
        best_subset: np.ndarray,
        history: List[float],
        config_used: Dict[str, Any],
        best_accuracy: float = 0.0  # <-- NEW: Added accuracy parameter
) -> None:
    """Saves the output of a single evolutionary run to results/{algo_name}/run_{run_id}.json"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    target_dir = base_dir / "results" / algo_name.lower()

    # Ensure the directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "run_id": run_id,
        "seed_used": seed,
        "best_fitness": float(best_fitness),
        "best_accuracy": float(best_accuracy), # <-- NEW: Saving it to the file
        "features_selected": int(best_subset.sum()),
        "selected_mask": best_subset.astype(int).tolist(),
        "convergence_history": history,
        "hyperparameters": config_used
    }

    file_path = target_dir / f"run_{run_id:02d}.json"

    with open(file_path, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"  -> Saved results to {file_path.relative_to(base_dir)}")