from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from time import time

import numpy as np
import rasterio
from tqdm.auto import trange

from ftcnn.io import clear_directory, collect_files_with_suffix
from ftcnn.utils import _WRITE_LOCK, NUM_CPU, TQDM_INTERVAL

from . import create_window, write_raster
from .conversion import raster_to_png


def process_raster_to_png_conversion(
    source_dir, dest_dir, *, recurse=True, preserve_dir=True, clear_dir=True, leave=True
):
    """
    Converts all TIFF files in the source directory (and subdirectories if specified) into PNG format
    and saves them to the destination directory, maintaining directory structure if needed.

    This function handles the following tasks:
    - Recursively collects all `.tif` files from the source directory.
    - Converts each TIFF file to PNG.
    - Optionally preserves the directory structure in the destination directory.
    - Optionally clears the destination directory before saving new files.

    Parameters:
        src_dir (PathLike): Source directory containing TIFF files to convert.
        dest_dir (PathLike): Destination directory to save converted PNG files.
        recurse (bool): Whether to recurse into subdirectories (default is True).
        preserve_dir (bool): Whether to preserve directory structure in destination (default is True).
        clear_dir (bool): Whether to clear the destination directory before saving (default is True).
        leave (bool): Whether to leave the progress bar displayed when the process is complete (default is True).

    Returns:
        dict: A dictionary mapping the original TIFF filenames (without extension) to their respective
              source TIFF and converted PNG file paths.

    Raises:
        Any exception raised by file reading/writing or conversion process.
    """
    file_map = {}

    source_dir = Path(source_dir).absolute()
    dest_dir = Path(dest_dir).absolute()

    src_paths = collect_files_with_suffix(".tif", source_dir, recurse=recurse)
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    elif clear_dir:
        clear_directory(dest_dir)

    def __exec__(path):
        if preserve_dir:
            relpath = path.relative_to(source_dir)
            dest_path = dest_dir / relpath.with_suffix(".png")
        else:
            dest_path = dest_dir / f"{path.name}.png"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        lock_id = _WRITE_LOCK.acquire()
        _ = raster_to_png(path, dest_path)
        _WRITE_LOCK.free(lock_id)
        file_map[path.stem] = {"tif": path, "png": dest_path}

    pbar = trange(len(src_paths), desc="Converting TIFF to PNG", leave=leave)
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = [executor.submit(__exec__, path) for path in src_paths]
        for _ in as_completed(futures):
            pbar.update()

    if leave:
        pbar.set_description("Complete")
    pbar.close()

    return file_map


def tile_raster_and_convert_to_png(source_path, *, tile_size):
    """
    Tiles a GeoTIFF file into smaller sections and converts each section into PNG images.

    This function:
    - Reads the input GeoTIFF file and extracts its EPSG code (coordinate reference system).
    - Creates tiles of the specified size from the GeoTIFF.
    - Converts each tile to a PNG image.
    - Returns a list of the PNG images and the EPSG code of the original GeoTIFF.

    Parameters:
        source_path (PathLike): Path to the GeoTIFF file to be tiled and converted.
        tile_size (tuple[int, int]): Size of the tiles (in pixels) to create from the GeoTIFF.

    Returns:
        tuple: A tuple containing:
            - A list of tuples, each containing a PNG image (as a NumPy array) and the coordinates of the tile.
            - The EPSG code of the original GeoTIFF.

    Raises:
        AttributeError: If the GeoTIFF file does not contain an EPSG code (CRS identifier).
        Any exception raised by file reading or conversion process.
    """
    epsg_code = None
    with rasterio.open(source_path) as src:
        if src.crs.is_epsg_code:
            # Returns a number indicating the EPSG code
            epsg_code = src.crs.to_epsg()

    images = []
    tiles = create_raster_tiles(source_path, tile_size=tile_size, crs=src.crs)

    for tiff, coords in tiles:
        image = raster_to_png(tiff)
        if image.max() != float("nan"):
            images.append((image, coords))
    return images, epsg_code


def create_raster_tiles(
    source_path: PathLike,
    tile_size: tuple[int, int],
    crs: str | None = None,
    output_dir: PathLike | None = None,
    exist_ok: bool = False,
    leave: bool = False,
) -> list[np.ndarray]:
    """
    Creates raster tiles from a source raster and saves them to disk (or returns them as arrays).

    Parameters:
        source_path (PathLike): The file path to the source raster.
        tile_size (tuple[int, int]): The size of the tiles (width, height).
        crs (str | None, optional): The coordinate reference system to use. Defaults to None.
        output_dir (PathLike | None, optional): Directory where the tiles should be saved. Defaults to None.
        exist_ok (bool, optional): If False, raises an error if tile files already exist. Defaults to False.
        leave (bool, optional): If True, leaves the progress bar description when finished. Defaults to False.

    Returns:
        list[np.ndarray]: A list of tile arrays (with corresponding metadata) or an empty list if no tiles are created.

    Example:
        >>> create_raster_tiles("input.tif", tile_size=(256, 256), output_dir="tiles")
    """
    source_path = Path(source_path)
    width, height = tile_size
    tiles = []

    with rasterio.open(source_path, crs=crs) as src:
        bounds = src.bounds
        rmin, cmin = src.index(bounds.left, bounds.top)
        rmax, cmax = src.index(bounds.right, bounds.bottom)
        rmin, rmax = min(rmin, rmax), max(rmin, rmax)
        cmin, cmax = min(cmin, cmax), max(cmin, cmax)

        if width <= 0 or height <= 0:
            return []

        total_updates = (rmax // height) * (cmax // width)
        updates = 0
        start = time()
        pbar = trange(total_updates, desc=f"Processing {source_path.name}", leave=leave)

        for row in range(rmin, rmax, height):
            for col in range(cmin, cmax, width):
                # Generate tile data and save if output directory is specified
                tile_output_path = None
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    tile_output_path = (
                        output_dir / f"{source_path.stem}_tile_{row}_{col}.tif"
                    )
                    if tile_output_path.exists() and not exist_ok:
                        raise FileExistsError(
                            f"File '{tile_output_path}' already exists"
                        )
                    output_dir.mkdir(parents=True, exist_ok=True)

                tile_window = create_window(
                    col,
                    row,
                    min(width, cmax - col),
                    min(height, rmax - row),
                )
                tile_data = src.read(1, window=tile_window)
                tile_data[tile_data == src.nodata] = np.NaN

                # Skip empty or irrelevant tiles
                if (
                    tile_data.shape[0] == 0
                    or tile_data.shape[1] == 0
                    or tile_data.max() == 0
                ):
                    continue

                tile_transform = src.window_transform(tile_window)
                lock_id = _WRITE_LOCK.acquire()
                tiles.append(
                    (
                        write_raster(
                            tile_data,
                            transform=tile_transform,
                            meta=src.meta.copy(),
                            output_path=tile_output_path,
                        ),
                        src.xy(row, col),
                    )
                )
                _WRITE_LOCK.free(lock_id)
                if time() - start >= TQDM_INTERVAL:
                    pbar.update()
                    updates += 1
                    start = time()
    pbar.close()
    return tiles
