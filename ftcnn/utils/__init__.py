from multiprocessing import cpu_count
from pathlib import Path

from .lock import Lock

TQDM_INTERVAL = 1 / 100
FTCNN_TMP_DIR = Path("/tmp", "ftcnn")
NUM_CPU = cpu_count()

FTCNN_TMP_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["Lock"]

_WRITE_LOCK = Lock()
