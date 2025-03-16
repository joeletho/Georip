from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

from .lock import Lock

TQDM_INTERVAL = 1 / 100
GEORIP_TMP_DIR = Path("/tmp", "georip")
NUM_CPU = min(8, cpu_count())

GEORIP_TMP_DIR.mkdir(parents=True, exist_ok=True)

StrPathLike = Union[str, Path]

__all__ = ["Lock"]

_WRITE_LOCK = Lock()
