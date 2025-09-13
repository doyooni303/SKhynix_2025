import yaml
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_config(config_path):
    """Load YAML configuration files"""
    configs = {}

    # Load all config files
    config_files = ["dataset", "model", "training"]

    for file in config_files:
        with open(f"configs/{file}.yaml", "r") as f:
            config = yaml.safe_load(f)
            configs.update(config)

    return configs


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
