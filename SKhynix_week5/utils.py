import os
import torch
import numpy as np
import yaml
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def load_config(config_dir: str = "configs") -> Dict:
    """YAML 설정 파일들을 통합하여 로드"""
    configs = {}
    config_files = ["dataset", "model", "training"]

    for file in config_files:
        config_path = os.path.join(config_dir, f"{file}.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            configs.update(config)

    return configs


def set_random_seeds(seed: int = 42):
    """재현성을 위한 랜덤 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str = "training.log"):
    """로깅 설정 및 global logger 설정"""
    global logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,  # 기존 핸들러 제거 후 새로 설정
    )

    logger = logging.getLogger(__name__)
    return logger
