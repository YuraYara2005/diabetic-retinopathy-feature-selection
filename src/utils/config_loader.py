import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_name: str) -> Dict[str, Any]:
    """Loads a YAML configuration file from the config/ directory."""
    # This automatically finds the root folder so you don't get path errors
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_path = base_dir / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)