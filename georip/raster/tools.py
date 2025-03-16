from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import ArgumentError
from multiprocessing.queues import Queue
from pathlib import Path
from time import time
from typing import Callable

import numpy as np
import rasterio
from tqdm.auto import trange

from georip.geometry.polygons import create_tile_polygon
from georip.io import clear_directory, collect_files_with_suffix
from georip.raster.utils import get_rows_cols_min_max_bounds
from georip.utils import NUM_CPU, TQDM_INTERVAL, StrPathLike

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
        src_dir (StrPathLike): Source directory containing TIFF files to convert.
        dest_dir (StrPathLike): Destination directory to save converted PNG files.
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
            # Check that this is not a file (may or may not exist)
            if len(relpath.suffix) > 0:
                relpath = relpath.parent
            dest_path = dest_dir / relpath

        dest_path = (dest_dir / path.stem).with_suffix(".png")

        _ = raster_to_png(path, dest_path)
        file_map[path.stem] = {"tif": path, "png": dest_path}

    pbar = trange(
        len(src_paths),
        desc=f"Converting {len(src_paths)} TIFF images to PNG format",
        leave=leave,
    )
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = [executor.submit(__exec__, path) for path in src_paths]
        for _ in as_completed(futures):
            pbar.update()

    if leave:
        num_pngs = len(collect_files_with_suffix(".png", dest_dir, recurse=True))
        pbar.set_description(f"Complete. Converted {num_pngs} images")
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
        source_path (StrPathLike): Path to the GeoTIFF file to be tiled and converted.
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

    _, tiles = create_raster_tiles(source_path, tile_size=tile_size, crs=src.crs)

    for tile, coords in tiles:
        image = raster_to_png(tile)
        if image.max() != float("nan"):
            images.append((image, coords))

    return images, epsg_code


def create_raster_tiles(
    source_path: StrPathLike,
    tile_size: tuple[int, int],
    crs: str | None = None,
    output_dir: StrPathLike | None = None,
    exist_ok: bool = False,
    leave: bool = False,
    filter_geometry: Callable | None = None,
    window: rasterio.windows.Window | None = None,
    stride: int | tuple[int, int] | None = None,
    callback_queue: Queue | None = None,
) -> tuple[Path, list[np.ndarray]]:
    """
    Creates raster tiles from a source raster and saves them to disk (or returns them as arrays).

    Parameters:
        source_path (StrPathLike): The file path to the source raster.
        tile_size (tuple[int, int]): The size of the tiles (width, height).
        crs (str | None, optional): The coordinate reference system to use. Defaults to None.
        output_dir (StrPathLike | None, optional): Directory where the tiles should be saved. Defaults to None.
        exist_ok (bool, optional): If False, raises an error if tile files already exist. Defaults to False.
        leave (bool, optional): If True, leaves the progress bar description when finished. Defaults to False.
        filter_geometry (Callable | None): A filter function to exclude tiles based on geometry.
        window (rasterio.windows.Window | None): Optional window to process a subset of the raster. Defaults to None.

    Returns:
        list[np.ndarray]: A list of tile arrays (with corresponding metadata) or an empty list if no tiles are created.
    """
    source_path = Path(source_path)
    width, height = tile_size
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be valid positive integers")
    tiles = []

    if stride is None:
        row_stride = height
        col_stride = width
    elif isinstance(stride, tuple):
        match (len(stride)):
            case 1:
                x, y = int(stride[0]), int(stride[0])
            case 2:
                x, y = int(stride[0]), int(stride[1])
            case _:
                raise ArgumentError(
                    f"Invalid stride type with tuple of length {len(stride)}"
                )

        row_stride = y
        col_stride = x
    elif isinstance(stride, int):
        if stride < 0:
            raise ArgumentError("Stride must be greater than or equal to 0")
        row_stride = col_stride = stride
    else:
        raise ArgumentError("Stride must be a None, int or tuple(int,int) value")

    with rasterio.open(source_path, crs=crs) as src:
        if window is None:
            (rmin, rmax), (cmin, cmax) = get_rows_cols_min_max_bounds(src)
        else:
            rmin, rmax = int(window.row_off), int(window.row_off + window.height)
            cmin, cmax = int(window.col_off), int(window.col_off + window.width)

        total_updates = ((rmax - rmin) // row_stride) * ((cmax - cmin) // col_stride)
        updates = 0
        start = time()
        pbar = None
        if callback_queue is None:
            pbar = trange(
                total_updates, desc=f"Processing {source_path.name}", leave=leave
            )

        def update_pbar():
            nonlocal pbar, start, updates, total_updates

            if time() - start >= TQDM_INTERVAL * NUM_CPU:
                start = time()
                if callback_queue is None:
                    pbar.update()
                else:
                    callback_queue.put((source_path, 1, total_updates))
            else:
                updates += 1

        meta = src.meta.copy()
        for row in range(rmin, rmax, row_stride):
            for col in range(cmin, cmax, col_stride):
                tile_window = create_window(
                    col,
                    row,
                    min(width, cmax - col),
                    min(height, rmax - row),
                )

                # Check for nodata-only tiles before proceeding
                mask = src.read_masks(1, window=tile_window)
                if np.all(mask == src.nodata):
                    update_pbar()
                    continue

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

                if filter_geometry:
                    window_bbox = create_tile_polygon(src, tile_window)
                    keep_geom = filter_geometry(source_path, window_bbox)
                    if not isinstance(keep_geom, bool):
                        raise ValueError(
                            "'filter_geometry' callback return expects a True or False"
                        )
                    if not keep_geom:
                        update_pbar()
                        continue

                bands = 1 if src.count == 1 else list(range(1, src.count + 1))
                tile_data = src.read(bands, window=tile_window, masked=True)

                if tile_data.size == 0:
                    continue

                tile_data[tile_data == src.nodata] = np.NaN

                tile_transform = src.window_transform(tile_window)
                # lock_id = _WRITE_LOCK.acquire()
                tiles.append(
                    (
                        write_raster(
                            tile_data,
                            transform=tile_transform,
                            meta=meta,
                            output_path=tile_output_path,
                            bands=bands,
                        ),
                        src.xy(tile_window.row_off, tile_window.col_off),
                    )
                )
                # _WRITE_LOCK.free(lock_id)
                update_pbar()

    if callback_queue is None:
        pbar.update(total_updates - updates + 1)
        pbar.close()
    else:
        callback_queue.put((source_path, total_updates - updates, total_updates))
    return source_path, tiles
