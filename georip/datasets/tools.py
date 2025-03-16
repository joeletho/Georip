from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import Manager
from multiprocessing.queues import Empty
from pathlib import Path
from time import sleep
from typing import Callable

import geopandas as gpd
import pandas as pd
from tqdm.auto import trange

from georip.datasets.utils import (
    TMP_FILE_PREFIX,
    StrPathLike,
    _filter_geometry_caller,
    init_dataset_filepaths,
    remove_unused_tiles,
)
from georip.geoprocessing.mapping import map_geometries_by_year_span
from georip.geoprocessing.processing import preprocess_ndvi_shapefile
from georip.geoprocessing.utils import translate_xy_coords_to_index
from georip.io import (
    clear_directory,
    collect_files_with_suffix,
    save_as_csv,
    save_as_gpkg,
    save_as_shp,
)
from georip.raster.tools import create_raster_tiles, process_raster_to_png_conversion
from georip.utils import NUM_CPU


def preprocess_ndvi_difference_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir: StrPathLike,
    years: tuple[int, int] | None = None,
    image_directory_column: str = "dirpath",
    image_filename_column: str = "filename",
    start_year_column: str = "start_year",
    end_year_column: str = "end_year",
    clean_dest: bool = False,
) -> list[Path]:
    """
    Preprocesses the NDVI difference dataset by selecting images based on the years and paths from the provided GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing metadata for the NDVI images.
        output_dir (StrPathLike): The directory to store the output data.
        years (tuple[int, int] | None, optional): A tuple of years to filter images. Defaults to None (all years).
        img_path_col (str, optional): The column containing the image file paths. Defaults to 'path'.
        start_year_col (str, optional): The column for the start year. Defaults to 'StartYear'.
        end_year_col (str, optional): The column for the end year. Defaults to 'EndYear'.
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.

    Returns:
        list[Path]: A list of paths to the selected NDVI images.

    Example:
        img_paths = preprocess_ndvi_difference_dataset(gdf, "output_dir", years=(2015, 2020))
    """

    def collect_image_paths(df):
        directories = df[image_directory_column]
        filenames = df[image_filename_column]
        return (
            Path(dir_path) / file_name
            for dir_path, file_name in zip(directories, filenames)
        )

    if years is None:
        image_paths = collect_image_paths(gdf)
    else:
        filtered_gdf = gdf[
            (gdf[start_year_column] == int(years[0]))
            & (gdf[end_year_column] == int(years[1]))
        ]
        image_paths = collect_image_paths(filtered_gdf)

    image_paths = list(set(image_paths))

    if len(image_paths) == 0:
        raise Exception("Could not find images")

    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    if clean_dest:
        clear_directory(output_dir)

    return image_paths


def multiprocess_create_raster_tiles(
    gdf: gpd.GeoDataFrame,
    target_imgs: list[Path],
    output_dir: Path,
    tile_size: int | tuple[int, int],
    *,
    exist_ok: bool,
    filter_geometry: Callable,
    stride: int | tuple[int, int] | None = None,
    num_workers: int = NUM_CPU,
):
    """
    Creates raster tiles from a list of input images using multiprocessing.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing spatial reference information.
        target_imgs (list[Path]): A list of file paths to the raster images to be processed.
        output_dir (Path): The directory where output tiles will be saved.
        tile_size (int | tuple[int, int]): The size of each tile in pixels.
        exist_ok (bool): Whether to overwrite existing tiles if they already exist.
        filter_geometry (Callable): A function used to filter geometries within the tiles.
        num_workers (int): Number of parallel processes to use (default: NUM_CPU).

    Yields:
        Path: The file path of each completed raster tile.

    Explanation:
        - Uses multiprocessing to create raster tiles from a set of input images.
        - A progress bar is managed for each image using a dictionary (`pbar_map`).
        - Each image is processed in a separate worker process using `ProcessPoolExecutor`.
        - The `progress_queue` tracks tile generation progress and updates the progress bars.
        - Completed tiles are yielded one by one, and exceptions from worker processes are handled.
    """
    pbar_map = dict()

    def update_pbar(pbar_map, path: Path, updates: int, total_updates: int):
        """
        Callback to update the progress bar for a specific image.
        """
        if not pbar_map.get(path.stem):
            return
        pbar = pbar_map[path.stem]["pbar"]
        if not pbar_map[path.stem]["initialized"]:
            pbar.reset(total_updates)
            pbar.set_description(f"Processing {path.stem}")
            pbar_map[path.stem]["initialized"] = True
        pbar.update(updates)

    def close_pbar(pbar_map, path: Path):
        pbar_group = pbar_map.get(path.stem)
        if pbar_group is not None:
            pbar = pbar_group["pbar"]
            pbar.update(pbar.total)
            pbar.close()
            pbar_map.pop(path.stem)

    with Manager() as manager:
        progress_queue = manager.Queue()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            size = tile_size if tile_size[0] is not None else None
            for path in target_imgs:
                pbar_map[path.stem] = {
                    "pbar": trange(
                        1, desc=f"Processing {path.name} (Queued)", leave=False
                    ),
                    "initialized": False,
                }

                futures.append(
                    executor.submit(
                        create_raster_tiles,
                        path,
                        crs=gdf.crs,
                        tile_size=size,
                        stride=stride,
                        output_dir=output_dir / path.stem,
                        exist_ok=exist_ok,
                        filter_geometry=filter_geometry,
                        callback_queue=progress_queue,
                    )
                )

            while futures:
                done_futures = [future for future in futures if future.done()]
                for future in done_futures:
                    exception = future.exception()
                    if exception:
                        raise exception
                    path = future.result()[0]
                    yield path
                    close_pbar(pbar_map, path)
                    futures.remove(future)

                try:
                    while True:
                        if any([future for future in futures if future.done()]):
                            break
                        path, updates, total = progress_queue.get_nowait()
                        update_pbar(pbar_map, path, updates, total)
                except Empty:
                    pass

    for path in target_imgs:
        close_pbar(pbar_map, path)


def preprocess_ndvi_difference_rasters(
    gdf: gpd.GeoDataFrame,
    output_dir: StrPathLike,
    *,
    years: tuple[int, int] | None = None,
    image_directory_column: str = "dirpath",
    image_filename_column: str = "filename",
    start_year_column: str = "start_year",
    end_year_column: str = "end_year",
    geom_column: str = "geometry",
    filter_geometry: Callable | None = None,
    tile_size: int | tuple[int, int] | None = None,
    stride: int | tuple[int, int] | None = None,
    exist_ok: bool = False,
    clean_dest: bool = False,
    leave: bool = True,
    num_workers: int | None = None,
    preserve_fields: list[dict[str, str]] | None = None,
) -> gpd.GeoDataFrame:
    """
    Preprocesses the NDVI difference rasters by creating tiles and mapping geometry to the corresponding GeoTIFFs.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing metadata and geometry.
        output_dir (StrPathLike): The directory to store output tiles and results.
        years (tuple[int, int] | None, optional): The years to filter the images. Defaults to None (all years).
        img_path_col (str, optional): The column containing the image file paths. Defaults to 'path'.
        start_year_col (str, optional): The column for the start year. Defaults to 'start_year'.
        end_year_col (str, optional): The column for the end year. Defaults to 'end_year'.
        geom_col (str, optional): The column containing geometries in the GeoDataFrame. Defaults to 'geometry'.
        tile_size (int | tuple[int, int] | None, optional): The size of the tiles to create. Defaults to None (default tile size).
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to False.
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.
        leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
        num_workers (int | None, optional): The number of worker threads to use. Defaults to None (auto-detect).

    Returns:
        GeoDataFrame: The updated GeoDataFrame after preprocessing, including geometry mapping to the GeoTIFFs.

    Example:
        gdf = preprocess_ndvi_difference_rasters(gdf, "output_dir", years=(2015, 2020), tile_size=(256, 256))
    """
    target_imgs = preprocess_ndvi_difference_dataset(
        gdf,
        output_dir,
        years=years,
        image_directory_column=image_directory_column,
        image_filename_column=image_filename_column,
        start_year_column=start_year_column,
        end_year_column=end_year_column,
        clean_dest=clean_dest,
    )
    tiles_dir = Path(output_dir) / "tif"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_size = tile_size if tile_size is not None else (None, None)
    if not isinstance(tile_size, tuple):
        tile_size = (tile_size, tile_size)

    pbar = trange(
        len(target_imgs),
        desc=f"Creating GeoTIFF tiles of size {f'({tile_size[0]},{tile_size[1]})' if tile_size[0] is not None else 'Default'}",
        leave=leave,
    )

    num_workers = (
        num_workers if num_workers and isinstance(num_workers, int) else NUM_CPU
    )

    def multiprocess_tiles():
        nonlocal pbar
        for _ in multiprocess_create_raster_tiles(
            gdf,
            target_imgs,
            output_dir=tiles_dir,
            tile_size=tile_size,
            stride=stride,
            exist_ok=exist_ok,
            filter_geometry=filter_geometry,
            num_workers=num_workers,
        ):
            pbar.update()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(multiprocess_tiles)
        while True:
            if future.done():
                break
            sleep(1)
            pbar.refresh()

    pbar.reset(total=3)
    pbar.set_description("Mapping geometry by year span")
    pbar.update()

    mapped_gdfs = map_geometries_by_year_span(
        gdf,
        tiles_dir,
        start_year_column,
        end_year_column,
        preserve_fields=preserve_fields,
    )

    if not len(mapped_gdfs) or mapped_gdfs[0].empty:
        raise ValueError(
            f"Did not find geometies for years {gdf[start_year_column].iat[0]} to {gdf[end_year_column].iat[0]}"
        )
    gdf = gpd.GeoDataFrame(pd.concat(mapped_gdfs, ignore_index=True), crs=gdf.crs)
    gdf.set_geometry("geometry", inplace=True)

    pbar.set_description("Cleaning up...")
    pbar.update()

    num_tiles_before = len(
        [
            str(path)
            for path in collect_files_with_suffix(
                [".tif", ".tiff"], tiles_dir, recurse=True
            )
        ]
    )

    gdf = remove_unused_tiles(
        gdf,
        geom_column=geom_column,
        image_directory_column=image_directory_column,
        image_filename_column=image_filename_column,
    )

    num_tiles_after = len(
        [
            str(path)
            for path in collect_files_with_suffix(
                [".tif", ".tiff"], tiles_dir, recurse=True
            )
        ]
    )

    num_diff = num_tiles_before - num_tiles_after
    print(
        "Processed {0} images and saved to {1}".format(
            max(0, num_tiles_before - num_diff), tiles_dir
        )
    )

    pbar.update()
    pbar.close()

    return gpd.GeoDataFrame(
        gdf.drop_duplicates()
        .sort_values(by=[start_year_column, end_year_column])
        .reset_index(drop=True)
    )


def make_ndvi_difference_dataset(
    source_shp: StrPathLike,
    source_images_dir: StrPathLike,
    output_dir: StrPathLike,
    *,
    years: tuple[int, int] | None = None,
    region_col: str | list[str] = "region",
    start_year_col: str = "start_year",
    end_year_col: str = "end_year",
    geom_col: str = "geometry",
    tile_size: int | tuple[int, int] | None = None,
    stride: int | tuple[int, int] | None = None,
    clean_dest: bool = False,
    translate_xy: bool = True,
    exist_ok: bool = False,
    save_csv: bool = False,
    save_shp: bool = False,
    save_gpkg: bool = False,
    convert_to_png: bool = True,
    pbar_leave: bool = True,
    num_workers: int | None = None,
    preserve_fields: list[dict[str, str]] | None = None,
) -> tuple[str, gpd.GeoDataFrame]:
    """
    Creates an NDVI difference dataset by processing a shapefile and the associated NDVI image files.

    Parameters:
        source_shp (StrPathLike): Path to the source shapefile containing metadata and geometry.
        images_dir (StrPathLike): Directory containing the NDVI image files.
        output_dir (StrPathLike): The directory where the processed dataset will be saved.
        years (tuple[int, int] | None, optional): A tuple of years to filter the images. Defaults to None (all years).
        start_year_col (str, optional): The column for the start year. Defaults to 'start_year'.
        end_year_col (str, optional): The column for the end year. Defaults to 'end_year'.
        geom_col (str, optional): The column containing geometry data in the GeoDataFrame. Defaults to 'geometry'.
        tile_size (int | tuple[int, int] | None, optional): The size of the tiles. Defaults to None (default tile size).
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.
        translate_xy (bool, optional): Whether to convert the coordinates to an index. Defaults to True.
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to False.
        save_csv (bool, optional): Whether to save the dataset as a CSV file. Defaults to False.
        save_shp (bool, optional): Whether to save the dataset as a shapefile. Defaults to False.
        save_gpkg (bool, optional): Whether to save the dataset as a geopackage. Defaults to False.
        convert_to_png (bool, optional): Whether to convert GeoTIFF files to PNG format. Defaults to True.
        pbar_leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
        num_workers (int | None, optional): The number of worker threads to use. Defaults to None (auto-detect).

    Returns:
        tuple: A tuple containing:
            - GeoDataFrame: The updated GeoDataFrame after preprocessing, including geometry and metadata.
            - tuple: A tuple containing the meta directory, tiles directory, and output file name.

    Example:
        gdf, meta = make_ndvi_difference_dataset("source.shp", "images_dir", "output_dir", years=(2015, 2020))
    """
    filepaths = init_dataset_filepaths(
        source_shp=source_shp,
        source_images_dir=source_images_dir,
        output_dir=output_dir,
        exist_ok=exist_ok,
        save_csv=save_csv,
        save_shp=save_shp,
        save_gpkg=save_gpkg,
        clean_dest=clean_dest,
    )

    source_shp = filepaths["shapefile"]
    output_dir = filepaths["output_dir"]
    source_images_dir = filepaths["image_dir_src"]
    tiles_dir = filepaths["tiles_dir"]
    csv_dir = filepaths["csv_dir"]
    shp_dir = filepaths["shp_dir"]

    n_calls = 5
    n_calls += 1 if translate_xy else 0
    n_calls += 1 if convert_to_png else 0

    pbar = trange(
        n_calls,
        desc="Creating NDVI dataset - Preprocessing shapefile",
        leave=pbar_leave,
    )

    ds_name = source_shp.stem.replace(TMP_FILE_PREFIX, "")

    if years is not None:
        ds_name += f"_{years[0]}to{years[1]}"
    ds_name = Path(ds_name)

    if not preserve_fields:
        preserve_fields = []

    if "start_year" not in preserve_fields:
        preserve_fields.append({start_year_col: "start_year"})
    if "end_year" not in preserve_fields:
        preserve_fields.append({end_year_col: "end_year"})

    gdf = preprocess_ndvi_shapefile(
        source_shp,
        years=years,
        region_col=region_col,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        images_dir=source_images_dir,
        preserve_fields=preserve_fields,
    )
    pbar.update()

    preserve_fields = [
        field
        for field in preserve_fields
        if not (
            isinstance(field, dict)
            and (
                (start_year_col in field and field[start_year_col] == "start_year")
                or (end_year_col in field and field[end_year_col] == "end_year")
            )
        )
    ]

    preserve_fields.append("start_year")
    preserve_fields.append("end_year")

    start_year_col = "start_year"
    end_year_col = "end_year"

    if save_csv:
        save_as_csv(gdf, csv_dir / ds_name.with_suffix(".csv"))
    if save_shp:
        save_as_shp(
            gdf,
            shp_dir / ds_name.with_suffix(".shp"),
        )
    if save_gpkg:
        save_as_gpkg(
            gdf,
            shp_dir / ds_name.with_suffix(".gpkg"),
        )

    pbar.update()
    pbar.set_description("Creating NDVI dataset - Preprocessing GeoTIFFs")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            preprocess_ndvi_difference_rasters,
            gdf,
            tiles_dir,
            start_year_column=start_year_col,
            end_year_column=end_year_col,
            geom_column=geom_col,
            years=years,
            tile_size=tile_size,
            stride=stride,
            clean_dest=clean_dest,
            exist_ok=exist_ok,
            leave=False,
            num_workers=num_workers,
            preserve_fields=preserve_fields,
            filter_geometry=partial(
                _filter_geometry_caller,
                gdf=gdf,
                region_column=region_col,
                start_year_column=start_year_col,
                end_year_column=end_year_col,
            ),
        )
        while True:
            if future.done():
                gdf = future.result()
                break
            sleep(1)
            pbar.refresh()

    pbar.update()

    if save_csv or save_shp:
        ds_name = Path(f"{ds_name}_tiles_xy")
        if save_csv:
            save_as_csv(gdf, csv_dir / ds_name.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / ds_name.with_suffix(".shp"),
            )
        if save_gpkg:
            save_as_gpkg(
                gdf,
                shp_dir / ds_name.with_suffix(".gpkg"),
            )

    if translate_xy:
        pbar.set_description("Creating NDVI dataset - Translating xy coords to index")

        gdf = translate_xy_coords_to_index(gdf)
        pbar.update()

        pbar.set_description("Cleaning up...")
        gdf = remove_unused_tiles(gdf, geom_col, "dirpath", "filename")

        if save_csv or save_shp:
            ds_name = str(ds_name).replace("_xy", "_indexed")
            if not ds_name.endswith("_indexed"):
                ds_name += "_indexed"
            ds_name = Path(ds_name)
            if save_csv:
                save_as_csv(gdf, csv_dir / ds_name.with_suffix(".csv"))
            if save_shp:
                save_as_shp(
                    gdf,
                    shp_dir / ds_name.with_suffix(".shp"),
                )
            if save_gpkg:
                save_as_gpkg(
                    gdf,
                    shp_dir / ds_name.with_suffix(".gpkg"),
                )
        pbar.update()

    if convert_to_png:
        pbar.set_description("Creating NDVI dataset - Converting GeoTIFFs to PNGs")

        tif_png_file_map = process_raster_to_png_conversion(
            tiles_dir / "tif", tiles_dir / "png", leave=True
        )
        pbar.update()

        pbar.reset(total=len(gdf))
        pbar.set_description("Creating NDVI dataset - Mapping filepaths")

        for i, row in gdf.iterrows():
            tif_file = str(row["filename"])
            paths = tif_png_file_map.get(Path(tif_file).stem)
            if paths is not None:
                gdf.loc[i, "filename"] = paths["png"].name
                gdf.loc[i, "dirpath"] = paths["png"].parent
            pbar.update()

        if save_csv or save_shp:
            ds_name = Path(f"{ds_name}_as_png")
            if save_csv:
                save_as_csv(gdf, csv_dir / ds_name.with_suffix(".csv"))
            if save_shp:
                save_as_shp(
                    gdf,
                    shp_dir / ds_name.with_suffix(".shp"),
                )
            if save_gpkg:
                save_as_gpkg(
                    gdf,
                    shp_dir / ds_name.with_suffix(".gpkg"),
                )

        pbar.reset(total=n_calls)
        pbar.update(n_calls - 1)

    pbar.set_description("Cleaning up...")
    gdf = remove_unused_tiles(gdf, geom_col, "dirpath", "filename")

    pbar.update()
    pbar.set_description("Creating NDVI dataset - Complete")
    pbar.close()

    return str(ds_name), gdf
