from os import PathLike
from typing import Union

import geopandas as gpd

from ftcnn.geometry.polygons import flatten_polygons
from ftcnn.geospacial.mapping import map_metadata
from ftcnn.io.geospacial import load_shapefile


def preprocess_ndvi_shapefile(
    source_path: PathLike,
    *,
    years: None | tuple[int, int],
    region_col: str | list[str],
    start_year_col: str,
    end_year_col: str,
    images_dir: PathLike,
    preserve_fields: Union[
        list[Union[str, dict[str, str]]],
        dict[str, str],
        None,
    ] = None,
) -> gpd.GeoDataFrame:
    """
    Preprocesses a shapefile by flattening polygons, removing duplicates, and mapping metadata.

    Parameters:
        source_path (PathLike): The path to the input shapefile to be processed.
        years (tuple[int, int] | None): A tuple indicating the start and end year for filtering rows. If None, no filtering is applied.
        region_col (str): The column name representing the region in the shapefile.
        start_year_col (str): The column name representing the start year in the shapefile.
        end_year_col (str): The column name representing the end year in the shapefile.
        images_dir (PathLike): The directory containing image data for metadata mapping.
        filter_geometry (bool): Whether to filter out empty geometries.
        preserve_fields (list | dict | None): Fields to preserve, either as a list of strings/dictionaries
                                              or a dictionary mapping input to output names.

    Returns:
        gpd.GeoDataFrame: A processed GeoDataFrame with flattened polygons, unique geometries,
                          and mapped metadata.
    """
    gdf = load_shapefile(source_path)

    # If years are provided, filter the rows based on the start and end years.
    if years is not None:
        start_year, end_year = years
        start_year = int(start_year)
        end_year = int(end_year)
        gdf = gdf[(gdf[start_year_col] >= start_year) & (gdf[end_year_col] <= end_year)]

    gdf = flatten_polygons(gdf, group_by=[*region_col, start_year_col, end_year_col])

    gdf = map_metadata(
        gdf,
        images_dir=images_dir,
        region_column=region_col,
        start_year_column=start_year_col,
        end_year_column=end_year_col,
        preserve_fields=preserve_fields,
    )

    if not len(gdf):
        raise ValueError("Shapefile does not contain valid metadata")

    return gdf
