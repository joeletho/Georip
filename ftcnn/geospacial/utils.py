import re
import sys
from os import PathLike
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd
import shapely
from rasterio import rasterio
from shapely import MultiPolygon, Polygon

from ftcnn.geometry import PolygonLike
from ftcnn.geometry.polygons import get_polygon_points
from ftcnn.geospacial import DataFrameLike
from ftcnn.geospacial.conversion import (
    translate_polygon_index_to_xy,
    translate_polygon_xy_to_index,
)


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


def encode_default_classes(row: pd.Series) -> tuple[int, str]:
    """
    Encodes default classes based on the geometry of a row.

    Parameters:
        row (pd.Series): A row from a DataFrame, expected to contain a "geometry" field.

    Returns:
        tuple[int, str]: A tuple containing:
            - class_id (int): 0 for treatment, -1 for background.
            - class_name (str): "Treatment" or "Background".
    """
    geom = row.get("geometry")
    return (
        (0, "Treatment")
        if geom is not None and not geom.is_empty and geom.area > 1
        else (-1, "Background")
    )


def tokenize_region_and_years_from_series(
    series: pd.Series,
    region_column: str,
    start_year_column: str,
    end_year_column: str,
) -> dict[str, str | tuple[int, int]]:
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
    image_path: PathLike,
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


def encode_classes(
    df: DataFrameLike, encoder: Callable = encode_default_classes
) -> DataFrameLike:
    """
    Adds encoded class information to a DataFrame.

    Parameters:
        df (DataFrameLike): The input DataFrame containing data to be encoded.
        encoder (Callable): A function that encodes a row into class ID and class name.
                            Defaults to `encode_default_classes`.

    Returns:
        DataFrameLike: A copy of the DataFrame with added "class_id" and "class_name" columns.
    """
    columns = {"class_id": [], "class_name": []}
    for _, row in df.iterrows():
        id, name = encoder(row)
        columns["class_id"].append(id)
        columns["class_name"].append(name)
    df_encoded = df.copy()
    df_encoded.insert(0, "class_id", columns["class_id"])
    df_encoded.insert(1, "class_name", columns["class_name"])
    return df_encoded


def translate_xy_coords_to_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Translates XY coordinates to pixel indices for geometries in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame with a "path" column containing file paths
                                and a "geometry" column with Polygon geometries.

    Returns:
        gpd.GeoDataFrame: A copy of the GeoDataFrame with updated "geometry" containing pixel indices.
    """
    gdf = gdf.copy()
    for i, row in gdf.iterrows():
        if Path(str(row["path"])).exists() and isinstance(row["geometry"], Polygon):
            polygon = translate_polygon_xy_to_index(row["path"], row["geometry"])
            gdf.at[i, "geometry"] = Polygon(get_polygon_points(polygon))
    return gdf


def translate_index_coords_to_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Translates pixel indices to XY coordinates for geometries in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame with a "path" column containing file paths
                                and a "geometry" column with Polygon geometries.

    Returns:
        gpd.GeoDataFrame: A copy of the GeoDataFrame with updated "geometry" containing XY coordinates.
    """
    gdf = gdf.copy()
    for i, row in gdf.iterrows():
        if Path(str(row["path"])).exists() and isinstance(row["geometry"], Polygon):
            polygon = translate_polygon_index_to_xy(row["path"], row["geometry"])
            gdf.at[i, "geometry"] = Polygon(get_polygon_points(polygon))
    return gdf


def raster_contains_polygon(
    source_path: str | PathLike, polygon: PolygonLike | list[PolygonLike]
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
                matched_rows = gdf[
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
                return bool(matched_rows.intersects(geometry).any())
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
