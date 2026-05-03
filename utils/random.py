from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, *parts: Any) -> int:
    hasher = hashlib.sha256()
    hasher.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        hasher.update(b"::")
        hasher.update(str(part).encode("utf-8"))
    return int(hasher.hexdigest()[:8], 16)
