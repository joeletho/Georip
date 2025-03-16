from os import PathLike
from pathlib import Path

import numpy as np
import skimage.io as skio
from osgeo import gdal, gdal_array
from skimage.util import img_as_float

from georip.io.geoprocessing import open_geo_dataset


def raster_to_png(
    source_path: PathLike, out_path: PathLike | None = None
) -> np.ndarray:
    """
    Converts a raster image to a PNG format.

    Parameters:
        source_path (PathLike): Path to the source raster.
        out_path (PathLike | None, optional): If provided, saves the PNG image to the specified path.

    Returns:
        np.ndarray: The PNG image as a numpy array.

    Example:
        >>> raster_to_png("input.tif", out_path="output.png")
    """
    png_img = None
    data = raster_to_array_3d(source_path)  # Convert the raster to a 3D array (RGB)

    # Convert to uint8 and save as PNG if required
    png_img = (img_as_float(data) * 255).astype(np.uint8)
    if out_path is not None:
        skio.imsave(out_path, png_img, check_contrast=False)  # Save the image

    return png_img


def raster_to_array_3d(source_path: PathLike) -> np.ndarray:
    """
    Converts a raster image to a 3D numpy array (RGB format).

    Parameters:
        source_path (PathLike): Path to the source raster file.

    Returns:
        np.ndarray: The raster data as a 3D numpy array (height x width x 3).

    Example:
        >>> raster_to_array_3d("input.tif")
    """
    if isinstance(source_path, str) or isinstance(source_path, Path):
        src_ds = open_geo_dataset(source_path)
    else:
        src_ds = gdal_array.OpenArray(source_path)

    src_width = src_ds.RasterXSize
    src_height = src_ds.RasterYSize

    dest_ds = gdal.GetDriverByName("MEM").Create(
        "", src_width, src_height, 3, gdal.GDT_Float64
    )
    nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()

    # Read and process the raster data
    src_arr = src_ds.ReadAsArray()
    src_ds = None

    if nodata_value is not None:
        mask = src_arr != nodata_value
        src_arr = np.where(mask, src_arr, np.nan)

    valid_data = src_arr[~np.isnan(src_arr)]
    if valid_data.size > 0:
        src_arr = np.interp(src_arr, (valid_data.min(), valid_data.max()), (0, 1))
    else:
        src_arr = np.zeros_like(src_arr)

    # Write the processed data to the new dataset
    for raster_index in range(1, 4):
        dest_ds.GetRasterBand(raster_index).WriteArray(src_arr)

    dest_ds.FlushCache()
    data = dest_ds.ReadAsArray()
    dest_ds = None
    data = np.moveaxis(data, 0, -1)

    return data
