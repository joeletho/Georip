import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio import DatasetReader, rasterio
from shapely import MultiPolygon, Polygon
from tqdm.auto import trange

from ftcnn.geometry import PolygonLike
from ftcnn.geometry.polygons import get_polygon_points, is_sparse_polygon
from ftcnn.geospacial import DataFrameLike
from ftcnn.geospacial.conversion import (translate_polygon_index_to_xy,
                                         translate_polygon_xy_to_index)
from ftcnn.raster import create_window
from ftcnn.utils import StrPathLike


def collect_filepaths(df: DataFrameLike, column_name: str) -> list[str]:
    """
    Collects file paths from a specified column in a DataFrame.

    Parameters:
        df (DataFrameLike): The DataFrame containing file path data.
        column_name (str): The name of the column to extract file paths from.

    Returns:
        list[str]: A list of file paths extracted from the specified column.
    """
    return list(df.loc[:, column_name].values())


def tokenize_region_and_years_from_series(
    series: pd.Series,
    region_column: str,
    start_year_column: str,
    end_year_column: str,
) -> dict[str, tuple[int, int]]:
    region = series.get(region_column)
    startyear = series.get(start_year_column)
    endyear = series.get(end_year_column)
    if region is None:
        raise ValueError(f"Could not find region in '{region_column}'")
    if startyear is None:
        raise ValueError(f"Could not find start year in '{start_year_column}'")
    if endyear is None:
        raise ValueError(f"Could not find end year in '{end_year_column}'")
    return {"region": region, "years": (int(startyear), int(endyear))}


def parse_filename(
    series: pd.Series,
    region_column: str,
    start_year_column: str,
    end_year_column: str,
) -> str:
    """
    Constructs a filename based on specific fields in a pandas Series.

    Parameters:
        series (pd.Series): A pandas Series containing the fields "Subregion", "StartYear", and "EndYear".

    Returns:
        str: A constructed filename string in the format:
             "[Identifier]_<Expanded_>[StartYear]to[EndYear]_NDVI_Difference.tif".
    """
    region = str(series[region_column])
    startyear = str(series[start_year_column])
    endyear = str(series[end_year_column])

    years_part = "to".join([startyear, endyear])
    end_part = "NDVI_Difference.tif"

    filename = region
    last = filename[-1]
    if last.isdigit():
        filename += "_"
    elif last == "E":
        filename = "_".join([filename[:-1], "Expanded", ""])
    start_part = filename + years_part
    return "_".join([start_part, end_part])


def parse_region_and_years_from_path(
    image_path: StrPathLike,
) -> tuple[str, tuple[int, int]]:
    """
    Parses the region and years from the file path.

    Parameters:
        image_path: str
            Path to the image file.

    Returns:
        Tuple[str, Tuple[int, int]]:
            Region and start-end year range.
    """
    parts = Path(image_path).stem.split("_")
    region = parts[0]
    years = parts[1]
    if "extended" in years.lower():
        region = region + "E"
        years = parts[2]
    elif region[-2:].isnumeric():
        start = 0
        while start < len(region) and not region[start].isdigit():
            start += 1
        if start >= len(region):
            raise ValueError(f"Error parsing years from {image_path}")
        years = region[start:]
        region = region[:start]
    years = years.split("to")
    return region, (int(years[0]), int(years[1]))


def translate_xy_coords_to_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Translates XY coordinates to pixel indices for geometries in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame with "dirpath" and "filename" columns containing file directory
                                path and filename, and a "geometry" column with Polygon geometries.

    Returns:
        gpd.GeoDataFrame: A copy of the GeoDataFrame with updated "geometry" containing pixel indices.
    """
    gdf = gdf.copy()
    for i, row in gdf.iterrows():
        filepath = Path(row["dirpath"]) / row["filename"]
        if filepath.exists() and isinstance(row["geometry"], Polygon):
            polygon = translate_polygon_xy_to_index(filepath, row["geometry"])
            gdf.at[i, "geometry"] = Polygon(get_polygon_points(polygon))
    return gdf


def translate_index_coords_to_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Translates pixel indices to XY coordinates for geometries in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame with "dirpath" and "filename" columns containing file directory
                                path and filename, and a "geometry" column with Polygon geometries.

    Returns:
        gpd.GeoDataFrame: A copy of the GeoDataFrame with updated "geometry" containing XY coordinates.
    """
    gdf = gdf.copy()
    for i, row in gdf.iterrows():
        filepath = Path(row["dirpath"]) / row["filename"]
        if filepath.exists() and isinstance(row["geometry"], Polygon):
            polygon = translate_polygon_index_to_xy(filepath, row["geometry"])
            gdf.at[i, "geometry"] = Polygon(get_polygon_points(polygon))
    return gdf


def raster_contains_polygon(
    source_path: StrPathLike, polygon: PolygonLike | list[PolygonLike]
) -> bool:
    """
    Checks if any of the provided polygons (or multipolygons) are fully contained within the bounds of the raster.

    Parameters:
        source_path (str or Path): Path to the raster source.
        polygons (PolygonLike or list of PolygonLike): Polygon or list of polygons to check.

    Returns:
        bool: True if any polygon is fully contained within the raster bounds, False otherwise.
    """
    with rasterio.open(source_path) as src:
        left, bottom, right, top = src.bounds

    raster_bbox = shapely.box(left, bottom, right, top)

    if isinstance(polygon, (Polygon, MultiPolygon)):
        polygon = [polygon]

    for poly in polygon:
        if raster_bbox.contains(poly):
            return True

    return False


def gdf_intersects_region_year_geometry(
    gdf, *, filepath, region_column, start_year_column, end_year_column, geometry
) -> bool:
    """
    Checks if the filename stem of a given filepath matches any combination of region and year
    in a GeoDataFrame and intersects with the specified geometry.

    Parameters:
        gdf: A GeoDataFrame containing the region, start year, and end year data.
        filepath: Path-like object representing the file path to be checked.
        region_column: Column name(s) in the GeoDataFrame representing region names.
        start_year_column: Column name representing the starting year in the GeoDataFrame.
        end_year_column: Column name representing the ending year in the GeoDataFrame.
        geometry: A geometry object to check intersection with.

    Returns:
        bool: True if a matching region and year combination intersects with the geometry, otherwise False.

    Raises:
        Prints an error message to stderr if matching rows are found but no valid intersection is detected.
    """
    if not geometry.is_empty and geometry.area > 1:
        if not isinstance(region_column, list):
            region_column = [region_column]

        path_stem = Path(filepath).stem
        for _, row in gdf.iterrows():
            start_year = row.get(start_year_column)
            start_year = int(start_year) if start_year else None
            end_year = row.get(end_year_column)
            end_year = int(end_year) if end_year else None

            for region in region_column:
                region_name = row.get(region)
                if region_name is None:
                    continue

                if stem_contains_region_and_years(
                    path_stem, str(region_name), str(start_year), str(end_year)
                ):
                    matched_rows = gdf.loc[
                        (gdf[region] == region_name)
                        & (gdf[start_year_column] == start_year)
                        & (gdf[end_year_column] == end_year)
                    ]

                    if matched_rows.empty:
                        print(
                            "Error matching rows for",
                            region_name,
                            start_year,
                            end_year,
                            file=sys.stderr,
                        )
                        return False

                    intersection = matched_rows.intersection(geometry).explode()
                    if intersection.empty:
                        return False

                    # Check for undesireable polygons. If we find any, fail.
                    sparse_mask = intersection.apply(is_sparse_polygon)
                    # print("Sparse mask:", sparse_mask)
                    is_sparse = all(sparse_mask)
                    # NOTE: DEBUG
                    # print(
                    #     "is_sparse:",
                    #     is_sparse,
                    #     "geometry:",
                    #     geometry,
                    # )
                    return not is_sparse
    return False


def stem_contains_region_and_years(stem, region, start_year, end_year):
    """
    Checks if a file stem contains a given region name and a combination of start and end years.

    Parameters:
        stem: The file stem as a string.
        region: The region name to check for.
        start_year: The starting year to check for.
        end_year: The ending year to check for.

    Returns:
        bool: True if the stem contains the region and year combination, otherwise False.
    """
    stem = stem.strip().replace("_", " ")
    pattern = rf"(?=.*\b{re.escape(region)}(?:\s*|_*)).*"
    return bool(re.match(pattern, stem, re.IGNORECASE)) and stem_contains_years(
        stem, start_year, end_year
    )


def stem_contains_years(stem, start_year, end_year):
    """
    Checks if a file stem contains both the start year and end year in sequence.

    Parameters:
        stem: The file stem as a string.
        start_year: The starting year to check for.
        end_year: The ending year to check for.

    Returns:
        bool: True if the stem contains both the start and end years, otherwise False.
    """
    stem = stem.strip().replace("_", " ")
    pattern = rf"(?=.*{re.escape(str(start_year))}.*{re.escape(str(end_year))}.*).*"
    return bool(re.match(pattern, stem, re.IGNORECASE))


def filter_by_region_and_years_columns(
    gdf, region_column, start_year_column, end_year_column
):
    """
    Filters a GeoDataFrame to only contain unique rows based on region(s), start year, and end year.

    Parameters:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.
        region_column (str or list of str): The name(s) of the region column(s).
        start_year (str): The column name for the start year.
        end_year (str): The column name for the end year.

    Returns:
        gpd.GeoDataFrame: A filtered GeoDataFrame with unique rows based on region, start year, and end year.
    """
    # If region_column is a list, we combine all columns
    if isinstance(region_column, list):
        columns_to_check = region_column + [start_year_column, end_year_column]
    else:
        columns_to_check = [region_column, start_year_column, end_year_column]

    # Drop duplicates based on the selected columns
    gdf_filtered = gdf.drop_duplicates(subset=columns_to_check).copy()

    return gdf_filtered


def debug_print_geom_with_regions_and_years(
    *,
    gdf,
    region_column,
    regions,
    start_column,
    start,
    end_column,
    end,
    message,
):
    if not isinstance(region_column, list):
        region_column = [region_column]
    if not isinstance(regions, list):
        regions = [regions]

    print(
        message + "\n",
        [
            gdf.loc[
                gdf[region].isin(regions)
                & (gdf[start_column] == start)
                & (gdf[end_column] == end)
            ]
            for region in region_column
        ],
    )


def get_gdf_valid_geometry(gdf, geometry_column):
    return gpd.GeoDataFrame(
        gdf[(gdf[geometry_column].notnull() & ~gdf[geometry_column].is_empty)],
        crs=gdf.crs,
    )


def gdf_ndvi_validate_years_as_ints(gdf, start_year_column, end_year_column):
    """
    Validates and converts the specified year columns in a GeoDataFrame to integers, handling invalid values.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the year columns to validate and convert.
        start_year_column (str): The name of the column representing the start year.
        end_year_column (str): The name of the column representing the end year.

    Returns:
        GeoDataFrame: The modified GeoDataFrame with the year columns converted to integers.
    """
    gdf = gdf.copy()

    for col in [start_year_column, end_year_column]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
        if gdf[col].isna().any():
            print(
                f"Warning: Found invalid entries in column '{col}', dropping rows with invalid values."
            )
            gdf = gdf.loc[gdf[col].notna()]
        gdf[col] = gdf[col].astype(int)

    return gdf


def gdf_matches_image_crs(gdf, images: StrPathLike | list[StrPathLike]) -> bool:
    """
    Checks if the coordinate reference system (CRS) of a GeoDataFrame matches the CRS of one or more GeoTIFF images.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame with a defined CRS.
        images (StrPathLike | list[StrPathLike]): A single path or a list of paths pointing to GeoTIFF image files.

    Returns:
        bool: True if the CRS of the GeoDataFrame matches the CRS of all provided images; False otherwise.

    Raises:
        ValueError: If 'images' is an empty list or does not contain valid path strings or StrPathLike objects.

    Explanation:
        - The function accepts a single image path or a list of image paths. If a single path is provided, it is wrapped in a list for consistency.
        - Each image file is opened using `rasterio` to access its CRS.
        - The function returns False if any image CRS differs from the GeoDataFrame CRS.
        - If all images have matching CRS, the function returns True.
    """
    if not isinstance(images, list):
        images = [images]
    if not len(images):
        raise ValueError(
            "'images' must contain a str, StrPathLike, or list of either pointing to the path of GeoTIFF image(s)"
        )

    for path in images:
        with rasterio.open(path) as src:
            if src.crs != gdf.crs:
                return False
    return True


def gdf_set_crs_to_image(gdf: DataFrameLike, image: StrPathLike) -> None:
    with rasterio.open(image) as src:
        gdf.to_crs(src.crs, inplace=True)


def get_image_windows(
    image_path: Path, *, max_size: int = 4096
) -> list[rasterio.windows.Window]:
    """
    Generates windows for splitting a large raster image into smaller tiles.

    Parameters:
        image_path (Path): Path to the input raster image.
        max_size (int): Maximum size (width or height) for each subset.

    Returns:
        list[rasterio.windows.Window]: List of windows representing image subsets.
    """
    width = height = None
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
    if width <= max_size and height <= max_size:
        return [create_window(0, 0, width, height)]

    windows = []
    num_tiles_x = (width + max_size - 1) // max_size
    num_tiles_y = (height + max_size - 1) // max_size

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            window = create_window(
                i * max_size,
                j * max_size,
                min(max_size, width - i * max_size),
                min(max_size, height - j * max_size),
            )
            windows.append(window)
    return windows


def get_src_data_nodata_transform(src):
    nodata = src.nodata or (src.nodatavals[0] if src.nodatavals else None)
    if nodata is None:
        raise ValueError("Raster does not have a nodata value defined.")
    data = src.read(1, masked=True)
    transform = src.transform
    return data, nodata, transform


def clip_geometries_to_raster(
    gdf: gpd.GeoDataFrame,
    raster: StrPathLike | DatasetReader,
    geometry_column: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Clips geometries in a GeoDataFrame to exclude areas above nodata pixels in the raster,
    while ensuring that geometries outside the raster are left unchanged.

    Parameters:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame with geometries to be clipped.
        raster (StrPathLike | DatasetReader): Path to the raster image or an open rasterio dataset.
        geometry_column (str): Name of the column containing geometry.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with updated geometries clipped to non-nodata areas.
    """
    gdf = gdf.copy()
    nodata = None
    data = None

    if isinstance(raster, StrPathLike):
        with rasterio.open(raster) as src:
            data, nodata, transform = get_src_data_nodata_transform(src)
    elif isinstance(raster, rasterio.DatasetReader):
        data, nodata, transform = get_src_data_nodata_transform(raster)
    else:
        raise ValueError(f"Invalid `raster` datatype '{type(raster)}'")

    return clip_geometries_to_raster_by_parts(
        gdf, data, nodata, transform, geometry_column=geometry_column
    )


def clip_geometries_to_raster_by_parts(
    gdf, srcdata, srcnodata, srctransform, *, geometry_column="geometry"
):
    gdf = gdf.copy()
    mask_shapes = rasterio.features.shapes(
        srcdata, srcdata.mask, transform=srctransform
    )
    valid_polygons = [
        shapely.geometry.shape(geom)
        for geom, value in mask_shapes
        if value != srcnodata
    ]
    combined_polygons = shapely.unary_union(valid_polygons)

    def get_geom_or_intersection(geom):
        intersection = geom.intersection(combined_polygons)
        if not intersection.is_empty:
            return intersection
        return geom

    gdf[geometry_column] = gdf[geometry_column].apply(get_geom_or_intersection)
    return gdf


def clip_geometries_to_raster_parallel(
    gdf: gpd.GeoDataFrame,
    raster_path: StrPathLike,
    geometry_column: str = "geometry",
    num_workers: int = 4,
    max_rows: int = 100,
) -> gpd.GeoDataFrame:
    """
    Clips geometries in a GeoDataFrame to exclude areas above nodata pixels in the raster, in parallel.

    Parameters:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame with geometries to be clipped.
        raster_path (StrPathLike): Path to the raster image.
        geometry_column (str): Name of the column containing geometry.
        num_workers (int): Number of threads to use for parallel processing.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with updated geometries clipped to non-nodata areas.
    """
    chunk_size = len(gdf) // max_rows
    gdf_chunks = [gdf.iloc[i : i + chunk_size] for i in range(0, len(gdf), chunk_size)]

    updated_gdf_list = []

    pbar = trange(
        len(gdf_chunks), desc=f"Processing {Path(raster_path).name}", leave=False
    )

    data = nodata = transform = None
    with rasterio.open(raster_path, crs=gdf.crs) as src:
        data, nodata, transform = get_src_data_nodata_transform(src)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for gdf_chunk in gdf_chunks:
            futures.append(
                executor.submit(
                    clip_geometries_to_raster_by_parts,
                    gdf_chunk,
                    data,
                    nodata,
                    transform,
                    geometry_column=geometry_column,
                )
            )

        for future in as_completed(futures):
            updated_gdf_list.append(future.result())
            pbar.update()
    pbar.close()
    return gpd.GeoDataFrame(pd.concat(updated_gdf_list, ignore_index=True), crs=gdf.crs)


def clip_geometries_to_rasters(
    gdf: gpd.GeoDataFrame,
    raster_paths: list[StrPathLike],
    raster_path_column: str,
    geometry_column: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Clips geometries in a GeoDataFrame for each raster image in the raster_paths list.
    It filters the gdf based on the image column, processes each image, and returns a new GeoDataFrame
    containing the clipped geometries.

    Parameters:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame with geometries to be clipped.
        raster_paths (list[StrPathLike]): List of raster image paths.
        raster_path_column (str): The name of the column containing the image paths in gdf.
        geometry_column (str): The name of the geometry column in gdf.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with clipped geometries for each image.
    """
    gdf = gdf.copy()

    for raster_path in raster_paths:
        raster_gdf = gdf[
            gdf[raster_path_column].apply(lambda path: Path(path) == Path(raster_path))
        ]
        if raster_gdf.empty:
            continue
        raster_gdf = raster_gdf.set_geometry(geometry_column)
        clipped_gdf = clip_geometries_to_raster_parallel(
            raster_gdf, raster_path, geometry_column
        )
        # Update the original gdf
        gdf.loc[clipped_gdf.index, geometry_column] = clipped_gdf[geometry_column]
    return gdf
