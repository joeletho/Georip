import numpy as np
from numpy._typing import ArrayLike


def linterp(arr: ArrayLike, new_min: int | float, new_max: int | float):
    """
    Linearly interpolate (rescale) an array to a new range [new_min, new_max].

    This function maps the values in `arr` (within its original range) to a new range,
    preserving the relative positions of the values.

    Formula:
        result = (arr - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min

    Parameters:
        arr (ArrayLike): Input array-like object (e.g., list or NumPy array).
        new_min (int | float): The lower bound of the new range.
        new_max (int | float): The upper bound of the new range.

    Returns:
        np.ndarray: A NumPy array with values scaled to the new range [new_min, new_max].

    Example:
        >>> arr = [0, 5, 10]
        >>> linterp(arr, 0, 1)
        array([0. , 0.5, 1. ])

        >>> arr = np.array([10, 20, 30])
        >>> linterp(arr, -1, 1)
        array([-1.,  0.,  1.])

    Notes:
        - `old_min` and `old_max` are automatically computed as the minimum and maximum
          of the input array.
        - If all values in `arr` are identical (`old_min == old_max`), a division-by-zero
          error will occur.
    """
    old_min = np.min(arr)
    old_max = np.max(arr)
    return (arr - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
