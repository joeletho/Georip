from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

from .lock import Lock

TQDM_INTERVAL = 1 / 100
FTCNN_TMP_DIR = Path("/tmp", "ftcnn")
NUM_CPU = min(8, cpu_count())

FTCNN_TMP_DIR.mkdir(parents=True, exist_ok=True)

StrPathLike = Union[str, Path]

__all__ = ["Lock"]

_WRITE_LOCK = Lock()
