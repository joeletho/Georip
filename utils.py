import shutil
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import numpy as np


def get_cpu_count():
    return cpu_count()


class Lock:
    def __init__(self):
        self._locked = False

    def is_locked(self):
        return self._locked

    def lock(self):
        assert not self._locked and "Lock is already locked"
        self._locked = True

    def unlock(self):
        assert self._locked and "Lock is already unlocked"
        self._locked = False


def pathify(path: str | Path, *args) -> Path | List[Path]:
    if not isinstance(path, list):
        paths = [path]
    else:
        paths = path
    if len(args):
        paths.extend([Path(arg) for arg in args])
    for i, path in enumerate(paths):
        paths[i] = Path(path)
    if len(paths) > 1:
        return paths
    return paths[0]


def clear_directory(dir_path):
    shutil.rmtree(dir_path)
    Path(dir_path).mkdir(parents=True)


def linterp(image, new_min, new_max):
    """
    result = (image - old_min) x ((new_max - new_min) /(old_max - old_min)) + new_min
    """
    old_min = np.min(image)
    old_max = np.max(image)
    return (image - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min


def collect_files_with_suffix(suffix, dir_path, *, recurse=False):
    dir_path = Path(dir_path)
    files = []
    if not isinstance(suffix, list):
        suffix = [suffix]
    for path in dir_path.iterdir():
        if path.is_dir() and recurse:
            files.extend(collect_files_with_suffix(suffix, path, recurse=recurse))
        else:
            if path.suffix in suffix:
                files.append(path)
    return files
