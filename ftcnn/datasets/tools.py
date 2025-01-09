from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd
from tqdm.auto import trange

from ftcnn.datasets.utils import (
    TMP_FILE_PREFIX,
    cleanup_unused_tiles,
    init_dataset_filepaths,
)
from ftcnn.geospacial.mapping import map_geometries_by_year_span
from ftcnn.geospacial.processing import preprocess_ndvi_shapefile
from ftcnn.geospacial.utils import (
    gdf_intersects_region_year_geometry,
    translate_xy_coords_to_index,
)
from ftcnn.io import (
    clear_directory,
    collect_files_with_suffix,
    pathify,
    save_as_csv,
    save_as_gpkg,
    save_as_shp,
)
from ftcnn.raster.tools import create_raster_tiles, process_raster_to_png_conversion
from ftcnn.utils import NUM_CPU


def preprocess_ndvi_difference_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir: PathLike,
    years: tuple[int, int] | None = None,
    img_path_col: str = "path",
    start_year_col: str = "start_year",
    end_year_col: str = "end_year",
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
            start_year = int(years[0])
            end_year = int(years[1])
            return (
                df.loc[
                    (df[start_year_col] == start_year) & (df[end_year_col] == end_year),
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
    filter_geometry: Callable | None = None,
    tile_size: int | tuple[int, int] | None = None,
    exist_ok: bool = False,
    clean_dest: bool = False,
    leave: bool = True,
    num_workers: int | None = None,
    preserve_fields: list[str | dict[str, str]] | None = None,
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
    tiles_dir = output_dir

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
                tile_size=tile_size if tile_size[0] is not None else None,
                output_dir=tiles_dir / path.stem,
                exist_ok=exist_ok,
                filter_geometry=filter_geometry,
            )
            for path in target_imgs
        ]
        for future in as_completed(futures):
            exception = future.exception()
            if exception:
                raise exception
            pbar.update()

    mapped_gdfs = map_geometries_by_year_span(
        gdf,
        tiles_dir,
        start_year_col,
        end_year_col,
        preserve_fields=preserve_fields,
    )

    if not len(mapped_gdfs) or mapped_gdfs[0].empty:
        raise ValueError(
            f"Did not find geometies for years {gdf[start_year_col].iat[0]} to {gdf[end_year_col].iat[0]}"
        )
    gdf = gpd.GeoDataFrame(pd.concat(mapped_gdfs, ignore_index=True), crs=gdf.crs)
    gdf.set_geometry("geometry", inplace=True)

    pbar.update()

    num_tiles = len(
        [
            str(path)
            for path in collect_files_with_suffix(".tif", tiles_dir, recurse=True)
        ]
    )
    pbar.update()
    pbar.close()

    num_tiles -= len(cleanup_unused_tiles(gdf, geom_col, img_path_col).keys())
    print("Processed {0} images and saved to {1}".format(max(0, num_tiles), tiles_dir))

    return gpd.GeoDataFrame(
        gdf.drop_duplicates()
        .sort_values(by=[start_year_col, end_year_col])
        .reset_index(drop=True)
    )


def make_ndvi_difference_dataset(
    source_shp: str | PathLike,
    source_images_dir: str | PathLike,
    output_dir: str | PathLike,
    *,
    years: tuple[int, int] | None = None,
    region_col: str | list[str] = "region",
    start_year_col: str = "start_year",
    end_year_col: str = "end_year",
    geom_col: str = "geometry",
    tile_size: int | tuple[int, int] | None = None,
    clean_dest: bool = False,
    translate_xy: bool = True,
    exist_ok: bool = False,
    save_csv: bool = False,
    save_shp: bool = False,
    save_gpkg: bool = False,
    convert_to_png: bool = True,
    pbar_leave: bool = True,
    num_workers: int | None = None,
    preserve_fields: list[str | dict[str, str]] | None = None,
) -> tuple[str, gpd.GeoDataFrame]:
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

    source_shp = filepaths["source_shp"]
    output_dir = filepaths["output_dir"]
    source_images_dir = filepaths["source_images_dir"]
    tiles_dir = filepaths["tiles_dir"]
    csv_dir = filepaths["csv_dir"]
    shp_dir = filepaths["shp_dir"]

    n_calls = 4
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
        leave=False,
        num_workers=num_workers,
        preserve_fields=preserve_fields,
        filter_geometry=lambda filepath, geom: gdf_intersects_region_year_geometry(
            gdf,
            filepath=filepath,
            geometry=geom,
            region_column=region_col,
            start_year_column=start_year_col,
            end_year_column=end_year_col,
        ),
    )

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
        pbar.update()
        pbar.set_description("Creating NDVI dataset - Translating xy coords to index")

        gdf = translate_xy_coords_to_index(gdf)
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
    if convert_to_png:
        pbar.update()
        pbar.set_description("Creating NDVI dataset - Converting GeoTIFFs to PNGs")

        tif_png_file_map = process_raster_to_png_conversion(
            tiles_dir, tiles_dir.parent / "png-tiles", leave=False
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
    pbar.update()
    pbar.set_description("Creating NDVI dataset - Complete")
    pbar.close()

    return str(ds_name), gdf
