from pathlib import Path

import geopandas as gpd
from osgeo import gdal
from shapely import wkt

from georip.utils import StrPathLike


def load_geo_dataframe(path: StrPathLike) -> gpd.GeoDataFrame:
    """
    Load spatial data into a GeoDataFrame from various file formats.

    Parameters:
        path (StrPathLike): The file path to load data from. Supported formats: CSV, Shapefile, GeoPackage.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the loaded spatial data.

    Notes:
        - If the file is a CSV, it is assumed to contain a 'geometry' column in WKT format.
        - For other formats, it uses `geopandas.read_file` to load the data.

    Example:
        >>> gdf = load_geo_dataframe("data_folder/spatial_data.shp")
        >>> gdf = load_geo_dataframe("data_folder/wkt_data.csv")
    """
    if Path(path).suffix == ".csv":
        df = gpd.read_file(path)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gpd_df = gpd.GeoDataFrame(df)
    else:
        gpd_df = gpd.read_file(path)
    return gpd_df


def open_geo_dataset(path: StrPathLike) -> gdal.Dataset:
    """
    Open a geospatial dataset using GDAL.

    Parameters:
        path (StrPathLike): The file path to the geospatial dataset.

    Returns:
        gdal.Dataset | None: The GDAL Dataset object if successful, otherwise None.

    Notes:
        - GDAL supports various geospatial file formats (e.g., GeoTIFF, Shapefiles).
        - This function converts the input `Path` to a string if necessary.

    Example:
        >>> ds = open_geo_dataset("data_folder/raster_data.tif")
        >>> if ds:
        ...     print("Dataset loaded successfully.")
    """
    if not isinstance(path, str):
        path = str(path)
    return gdal.Open(path)


def load_shapefile(path: StrPathLike) -> gpd.GeoDataFrame:
    """
    Loads a shapefile or CSV file containing geometry data and returns it as a GeoDataFrame.

    This function handles the loading of both shapefiles and CSV files containing geometries.
    - If the file is a CSV, it expects the geometry column to be in Well-Known Text (WKT) format
      and will convert it into a proper geometry column.
    - If the file is a shapefile, it is directly loaded as a GeoDataFrame.

    Parameters:
        path (StrPathLike): Path to the shapefile or CSV file to load.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the data from the input file, with geometries
                           properly processed for either format.

    Raises:
        Any exception raised by GeoPandas' `read_file` method or WKT parsing (for CSVs).
    """
    if Path(path).suffix == ".csv":
        df = gpd.read_file(path)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df)
    else:
        gdf = gpd.read_file(path)
    return gdf
