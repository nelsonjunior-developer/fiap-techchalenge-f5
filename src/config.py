"""Global project configuration values for reproducibility and consistency."""

from __future__ import annotations

import random
from src.utils import get_logger


RANDOM_STATE: int = 42
SEED: int = RANDOM_STATE
_logger = get_logger(__name__)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Apply the global seed to standard Python RNG and numpy when available."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # Keep this helper dependency-free even when numpy is unavailable.
        pass

    _logger.info("Global seed configured | seed=%d", seed)
