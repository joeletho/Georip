import io
from os import PathLike
from typing import Any

import numpy as np
import rasterio
import rioxarray as rxr
from numpy.typing import ArrayLike
from rasterio.windows import Window


def open_raster(path, *, masked=True):
    """
    Opens a raster file and returns its data as an xarray.DataArray object, optionally applying a mask to handle nodata values.

    Parameters:
        path: PathLike
            The file path to the raster dataset.
        masked: bool, optional
            If True, applies a mask to the raster data to account for nodata values. Default is True.

    Returns:
        xarray.DataArray: The raster data as a squeezed xarray.DataArray, reducing any singleton dimensions.

    Raises:
        FileNotFoundError: If the specified raster file does not exist.
        rasterio.errors.RasterioIOError: If the file cannot be opened as a raster dataset.
    """
    return rxr.open_rasterio(path, masked=masked).squeeze()


def create_virtual_image(
    data: Any, *, bands: int | list[int], meta: dict, format: str = ".tif"
) -> np.ndarray:
    """
    Creates a virtual image from raw data in memory without writing it to disk.

    Parameters:
        data (Any): The image data to be written (e.g., a numpy array).
        bands (int | list[int]): The bands to be written from the image data.
        meta (dict): Metadata for the raster (e.g., transform, CRS).
        format (str, optional): The format for the virtual image. Defaults to ".tif".

    Returns:
        np.ndarray: The image data as a numpy array after reshaping.

    Example:
        >>> virtual_image = create_virtual_image(data, bands=1, meta=meta)
    """
    buffer = io.BytesIO()  # Create an in-memory buffer
    image = None

    # Use MemoryFile to write the data to the virtual image
    with rasterio.MemoryFile(buffer, ext=format) as mem:
        with mem.open(**meta) as dest:
            dest.write(data, bands)  # Write data to the virtual image
        with mem.open() as src:
            arr = src.read(bands)  # Read the bands back into an array
            image = arr.reshape(data.shape)  # Reshape to match the original data shape

    return image


def write_raster(
    data: np.ndarray,
    *,
    transform: rasterio.Affine,
    meta: dict,
    output_path: PathLike | None = None,
    bands: int | list[int] = 1,
) -> np.ndarray | None:
    """
    Writes raster data to a file or creates a virtual image if no file path is provided.

    Parameters:
        data (np.ndarray): The image data to be written.
        transform (rasterio.Affine): The affine transform for the raster.
        meta (dict): Metadata for the raster (e.g., CRS, dimensions).
        output_path (PathLike, optional): The file path where the raster will be saved. If None, a virtual image is created.
        bands (int | list[int], optional): The bands to be written. Defaults to 1.

    Returns:
        np.ndarray | None: If no output path is given, returns the created virtual image. Otherwise, returns None.

    Example:
        >>> write_raster(data, transform=affine, meta=meta, output_path="output.tif")
    """
    meta.update(
        {
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
        }
    )
    res: ArrayLike | None = None
    # If no output path is provided, create a virtual image
    if output_path is None:
        res = create_virtual_image(data, bands=bands, meta=meta)
    else:
        # Otherwise, write the raster data to the specified path
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(data, bands)

    return res


def create_window(col, row, width, height):
    return Window(col_off=col, row_off=row, width=width, height=height)
