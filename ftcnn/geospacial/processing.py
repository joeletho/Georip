from os import PathLike

import geopandas as gpd

from ftcnn.geometry.polygons import flatten_polygons
from ftcnn.geospacial.mapping import map_metadata
from ftcnn.io.geospacial import load_shapefile


def preprocess_shapefile(
    source_path: PathLike,
    start_year_col: str,
    end_year_col: str,
    images_dir: PathLike,
) -> gpd.GeoDataFrame:
    """
    Preprocesses a shapefile by flattening polygons, removing duplicates, and mapping metadata.

    Parameters:
        shpfile (PathLike): The path to the input shapefile to be processed.
        start_year_col (str): The column name representing the start year in the shapefile.
        end_year_col (str): The column name representing the end year in the shapefile.
        img_dir (PathLike): The directory containing image data for metadata mapping.

    Returns:
        gpd.GeoDataFrame: A processed GeoDataFrame with flattened polygons, unique geometries,
                          and mapped metadata.
    """
    # Load the shapefile into a GeoDataFrame.
    gdf = load_shapefile(source_path)
    crs = gdf.crs  # Store the original Coordinate Reference System (CRS).

    # Flatten polygons by grouping based on the start and end year columns.
    gdf = flatten_polygons(gdf, group_by=[start_year_col, end_year_col])

    # Remove duplicate geometries and records.
    gdf = gpd.GeoDataFrame(gdf.drop_duplicates())

    # Map metadata to the GeoDataFrame, preserving specific fields for start and end years.
    gdf = map_metadata(
        gdf,
        images_dir=images_dir,
        preserve_fields={"start_year": start_year_col, "end_year": end_year_col},
    )
    if not len(gdf):
        raise ValueError("Shapefile does not contain valid metadata")

    # Remove any additional duplicates after metadata mapping.
    return gpd.GeoDataFrame(gdf.drop_duplicates(), crs=crs)
