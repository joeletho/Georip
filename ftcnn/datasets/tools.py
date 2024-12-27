import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm, trange

from ftcnn.geospacial.mapping import map_geometry_to_geotiffs
from ftcnn.geospacial.processing import preprocess_shapefile
from ftcnn.geospacial.utils import translate_xy_coords_to_index
from ftcnn.io import (clear_directory, collect_files_with_suffix, pathify,
                      save_as_csv, save_as_gpkg, save_as_shp)
from ftcnn.raster.tools import (create_raster_tiles,
                                process_raster_to_png_conversion)
from ftcnn.utils import NUM_CPU


def preprocess_ndvi_difference_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir: PathLike,
    years: tuple[int, int] | None = None,
    img_path_col: str = "path",
    start_year_col: str = "StartYear",
    end_year_col: str = "EndYear",
    clean_dest: bool = False,
) -> list[Path]:
    """
    Preprocesses the NDVI difference dataset by selecting images based on the years and paths from the provided GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing metadata for the NDVI images.
        output_dir (PathLike): The directory to store the output data.
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
    if years is None:

        def all_images(df):
            images = df.loc[:, img_path_col].unique().tolist()
            return images

        get_filepaths = all_images
    else:

        def match_years(df):
            return (
                df.loc[
                    (df[start_year_col] == years[0]) & (df[end_year_col] == years[1]),
                    img_path_col,
                ]
                .unique()
                .tolist()
            )

        get_filepaths = match_years

    img_paths = pathify(get_filepaths(gdf))
    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    if len(img_paths) == 0:
        raise Exception("Could not find images")

    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    if clean_dest:
        clear_directory(output_dir)

    return img_paths


def preprocess_ndvi_difference_rasters(
    gdf: gpd.GeoDataFrame,
    output_dir: PathLike,
    *,
    years: tuple[int, int] | None = None,
    img_path_col: str = "path",
    start_year_col: str = "start_year",
    end_year_col: str = "end_year",
    geom_col: str = "geometry",
    tile_size: int | tuple[int, int] | None = None,
    exist_ok: bool = False,
    clean_dest: bool = False,
    ignore_empty_geom: bool = True,
    leave: bool = True,
    num_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Preprocesses the NDVI difference rasters by creating tiles and mapping geometry to the corresponding GeoTIFFs.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing metadata and geometry.
        output_dir (PathLike): The directory to store output tiles and results.
        years (tuple[int, int] | None, optional): The years to filter the images. Defaults to None (all years).
        img_path_col (str, optional): The column containing the image file paths. Defaults to 'path'.
        start_year_col (str, optional): The column for the start year. Defaults to 'start_year'.
        end_year_col (str, optional): The column for the end year. Defaults to 'end_year'.
        geom_col (str, optional): The column containing geometries in the GeoDataFrame. Defaults to 'geometry'.
        tile_size (int | tuple[int, int] | None, optional): The size of the tiles to create. Defaults to None (default tile size).
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to False.
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.
        ignore_empty_geom (bool, optional): Whether to ignore empty geometries. Defaults to True.
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
        years,
        img_path_col,
        start_year_col,
        end_year_col,
        clean_dest,
    )

    tile_size = tile_size if tile_size is not None else (None, None)
    if not isinstance(tile_size, tuple):
        tile_size = (tile_size, tile_size)

    pbar = trange(
        len(target_imgs) + 2,
        desc=f"Creating GeoTIFF tiles of size {f'({tile_size[0]},{tile_size[1]})' if tile_size[0] is not None else 'Default'}",
        leave=leave,
    )

    num_workers = num_workers if num_workers else NUM_CPU

    # Make the tiles for each GeoTIFF
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                create_raster_tiles,
                path,
                crs=gdf.crs,
                tile_size=tile_size,
                output_dir=output_dir / path.stem,
                exist_ok=exist_ok,
            )
            for path in target_imgs
        ]
        for _ in as_completed(futures):
            pbar.update()

    # Group rows by the years in which they span. We want to ensure we are only
    # applying geometry to rows which share the same start and end years.
    year_pairs = gdf[[start_year_col, end_year_col]].drop_duplicates()
    year_pairs = year_pairs.sort_values(by=[start_year_col, end_year_col])
    start_years = [int(year) for year in year_pairs[start_year_col].tolist()]
    end_years = [int(year) for year in year_pairs[end_year_col].tolist()]

    mapped_gdfs = []
    for start_year, end_year in zip(start_years, end_years):
        # Get the rows which match the start and end years
        target_years = gpd.GeoDataFrame(
            gdf[(gdf[start_year_col] == start_year) & (gdf[end_year_col] == end_year)]
        )
        gdf_mapped = map_geometry_to_geotiffs(
            target_years,
            output_dir,
        )
        # Insert the year columns into the mapped geodf
        gdf_mapped.insert(0, start_year_col, start_year)
        gdf_mapped.insert(1, end_year_col, end_year)

        mapped_gdfs.append(gdf_mapped)

    gdf = gpd.GeoDataFrame(pd.concat(mapped_gdfs, ignore_index=True), crs=gdf.crs)
    gdf.set_geometry("geometry", inplace=True)

    pbar.update()

    tile_paths = [
        str(path)
        for path in collect_files_with_suffix(".tif", output_dir, recurse=True)
    ]
    unused_tiles = []

    if ignore_empty_geom:
        # Remove any tiles that do not map to an image in the dataframe
        unused_tiles = gdf.loc[gdf[geom_col].is_empty, img_path_col].tolist()
        gdf = gpd.GeoDataFrame(gdf[~gdf[geom_col].is_empty].reset_index(drop=True))

        for path in tqdm(unused_tiles, desc="Cleaning up", leave=False):
            path = Path(path)
            parent = path.parent
            parent_parent = parent.parent
            if path.exists():
                os.remove(path)
            if parent.exists() and len(os.listdir(parent)) == 0:
                os.rmdir(parent)
                if parent_parent.exists() and len(os.listdir(parent_parent)) == 0:
                    os.rmdir(parent_parent)

    pbar.update()
    nfiles = max(0, len(tile_paths) - len(unused_tiles))
    pbar.set_description(
        "Processed {0} images and saved to {1}".format(nfiles, output_dir)
    )
    pbar.close()

    return gpd.GeoDataFrame(
        gdf.drop_duplicates()
        .sort_values(by=[start_year_col, end_year_col])
        .reset_index(drop=True)
    )


def make_ndvi_difference_dataset(
    source_shp: PathLike,
    images_dir: PathLike,
    output_dir: PathLike,
    *,
    years: tuple[int, int] | None = None,
    start_year_col: str = "start_year",
    end_year_col: str = "end_year",
    geom_col: str = "geometry",
    tile_size: int | tuple[int, int] | None = None,
    clean_dest: bool = False,
    xy_to_index: bool = True,
    exist_ok: bool = False,
    save_csv: bool = False,
    save_shp: bool = False,
    save_gpkg: bool = False,
    ignore_empty_geom: bool = True,
    tif_to_png: bool = True,
    pbar_leave: bool = True,
    num_workers: int | None = None,
) -> tuple[gpd.GeoDataFrame, tuple[Path, Path, Path]]:
    """
    Creates an NDVI difference dataset by processing a shapefile and the associated NDVI image files.

    Parameters:
        source_shp (PathLike): Path to the source shapefile containing metadata and geometry.
        images_dir (PathLike): Directory containing the NDVI image files.
        output_dir (PathLike): The directory where the processed dataset will be saved.
        years (tuple[int, int] | None, optional): A tuple of years to filter the images. Defaults to None (all years).
        start_year_col (str, optional): The column for the start year. Defaults to 'start_year'.
        end_year_col (str, optional): The column for the end year. Defaults to 'end_year'.
        geom_col (str, optional): The column containing geometry data in the GeoDataFrame. Defaults to 'geometry'.
        tile_size (int | tuple[int, int] | None, optional): The size of the tiles. Defaults to None (default tile size).
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.
        xy_to_index (bool, optional): Whether to convert the coordinates to an index. Defaults to True.
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to False.
        save_csv (bool, optional): Whether to save the dataset as a CSV file. Defaults to False.
        save_shp (bool, optional): Whether to save the dataset as a shapefile. Defaults to False.
        save_gpkg (bool, optional): Whether to save the dataset as a geopackage. Defaults to False.
        ignore_empty_geom (bool, optional): Whether to ignore empty geometries. Defaults to True.
        tif_to_png (bool, optional): Whether to convert GeoTIFF files to PNG format. Defaults to True.
        pbar_leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
        num_workers (int | None, optional): The number of worker threads to use. Defaults to None (auto-detect).

    Returns:
        tuple: A tuple containing:
            - GeoDataFrame: The updated GeoDataFrame after preprocessing, including geometry and metadata.
            - tuple: A tuple containing the meta directory, tiles directory, and output file name.

    Example:
        gdf, meta = make_ndvi_difference_dataset("source.shp", "images_dir", "output_dir", years=(2015, 2020))
    """
    source_shp, images_dir, output_dir = (
        Path(source_shp),
        Path(images_dir),
        Path(output_dir),
    )
    meta_dir = output_dir / "meta"
    csv_dir = meta_dir / "csv" / source_shp.stem
    shp_dir = meta_dir / "shp" / source_shp.stem

    if output_dir.exists() and clean_dest:
        clear_directory(output_dir)
    elif not output_dir.exists():
        output_dir.mkdir(parents=True)

    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=exist_ok)
    if save_shp or save_gpkg:
        shp_dir.mkdir(parents=True, exist_ok=exist_ok)

    tiles_dir = output_dir / "images" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=exist_ok)

    n_calls = 4
    n_calls += 1 if xy_to_index else 0
    n_calls += 1 if tif_to_png else 0

    pbar = trange(
        n_calls,
        desc="Creating NDVI dataset - Preprocessing shapefile",
        leave=pbar_leave,
    )

    gdf = preprocess_shapefile(
        source_shp,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        images_dir=images_dir,
    )
    pbar.update()

    start_year_col = "start_year"
    end_year_col = "end_year"

    output_fname = source_shp.stem
    if years is not None:
        output_fname += f"_{years[0]}to{years[1]}"
    output_fname = Path(output_fname)

    if save_csv:
        save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
    if save_shp:
        save_as_shp(
            gdf,
            shp_dir / output_fname.with_suffix(".shp"),
        )
    if save_gpkg:
        save_as_gpkg(
            gdf,
            shp_dir / output_fname.with_suffix(".gpkg"),
        )

    pbar.update()
    pbar.set_description("Creating NDVI dataset - Preprocessing GeoTIFFs")

    gdf = preprocess_ndvi_difference_rasters(
        gdf,
        tiles_dir,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        geom_col=geom_col,
        years=years,
        tile_size=tile_size,
        clean_dest=clean_dest,
        exist_ok=exist_ok,
        ignore_empty_geom=ignore_empty_geom,
        leave=False,
        num_workers=num_workers,
    )
    pbar.update()

    if save_csv or save_shp:
        output_fname = Path(f"{output_fname}_tiles_xy")
        if save_csv:
            save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / output_fname.with_suffix(".shp"),
            )
        if save_gpkg:
            save_as_gpkg(
                gdf,
                shp_dir / output_fname.with_suffix(".gpkg"),
            )

    if xy_to_index:
        pbar.update()
        pbar.set_description("Creating NDVI dataset - Translating xy coords to index")

        gdf = translate_xy_coords_to_index(gdf)
        if save_csv or save_shp:
            output_fname = str(output_fname).replace("_xy", "_indexed")
            if not output_fname.endswith("_indexed"):
                output_fname += "_indexed"
            output_fname = Path(output_fname)
            if save_csv:
                save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
            if save_shp:
                save_as_shp(
                    gdf,
                    shp_dir / output_fname.with_suffix(".shp"),
                )
            if save_gpkg:
                save_as_gpkg(
                    gdf,
                    shp_dir / output_fname.with_suffix(".gpkg"),
                )
    if tif_to_png:
        pbar.update()
        pbar.set_description("Creating NDVI dataset - Converting GeoTIFFs to PNGs")

        tif_png_file_map = process_raster_to_png_conversion(
            tiles_dir, output_dir / "images" / "png-tiles", leave=False
        )
        spbar = trange(len(gdf), desc="Mapping filepaths", leave=False)
        for i, row in gdf.iterrows():
            tif_file = str(row["filename"])
            paths = tif_png_file_map.get(Path(tif_file).stem)
            if paths is not None:
                gdf.loc[i, "filename"] = paths["png"].name
                gdf.loc[i, "path"] = paths["png"]
            spbar.update()
        spbar.close()
        if save_csv or save_shp:
            output_fname = Path(f"{output_fname}_as_png")
            if save_csv:
                save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
            if save_shp:
                save_as_shp(
                    gdf,
                    shp_dir / output_fname.with_suffix(".shp"),
                )
            if save_gpkg:
                save_as_gpkg(
                    gdf,
                    shp_dir / output_fname.with_suffix(".gpkg"),
                )
    pbar.update()
    pbar.set_description("Creating NDVI dataset - Complete")
    pbar.close()

    return gdf, (meta_dir, tiles_dir, output_fname)
