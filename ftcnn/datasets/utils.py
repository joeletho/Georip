import os
import time
from os import PathLike
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ftcnn.geospacial import DataFrameLike
from ftcnn.io import clear_directory, save_as_shp
from ftcnn.io.geospacial import load_shapefile
from ftcnn.utils import FTCNN_TMP_DIR

TMP_FILE_PREFIX = "tmp__"


def init_dataset_filepaths(
    *,
    source_shp: str | PathLike,
    source_images_dir: str | PathLike,
    output_dir: str | PathLike,
    save_csv: bool = True,
    save_shp: bool = True,
    save_gpkg: bool = True,
    clean_dest: bool = False,
    exist_ok: bool = False,
) -> dict[str, Path]:
    source_shp, source_images_dir, output_dir = (
        Path(source_shp),
        Path(source_images_dir),
        Path(output_dir),
    )
    meta_dir: Path = output_dir / "meta"
    csv_dir: Path = meta_dir / "csv" / source_shp.stem
    shp_dir: Path = meta_dir / "shp" / source_shp.stem

    if output_dir.exists() and clean_dest:
        clear_directory(output_dir)
    elif not output_dir.exists():
        output_dir.mkdir(parents=True)

    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=exist_ok)
    if save_shp or save_gpkg:
        shp_dir.mkdir(parents=True, exist_ok=exist_ok)

    tiles_dir: Path = output_dir / "images" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=exist_ok)
    return {
        "source_shp": Path(source_shp),
        "output_dir": Path(output_dir),
        "source_images_dir": Path(source_images_dir),
        "tiles_dir": Path(tiles_dir),
        "meta_dir": Path(meta_dir),
        "csv_dir": Path(csv_dir),
        "shp_dir": Path(shp_dir),
    }


def cleanup_unused_tiles(
    gdf: gpd.GeoDataFrame, geom_col: str, img_path_col: str
) -> gpd.GeoDataFrame:
    """
    Removes unused tiles (files and empty directories) based on a GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing geometry and image paths.
        geom_col (str): The column name in `gdf` that contains geometries.
        img_path_col (str): The column name in `gdf` that contains image paths.

    Returns:
        GeoDataFrame: The updated GeoDataFrame with non-empty geometries.
    """
    unused_tiles = []

    # Remove any tiles that do not map to an image in the dataframe
    unused_tiles = gdf.loc[gdf[geom_col].is_empty, img_path_col].unique().tolist()
    gdf = gpd.GeoDataFrame(gdf[~gdf[geom_col].is_empty].reset_index(drop=True))

    for path in unused_tiles:
        path = Path(path)
        parent = path.parent
        if path.exists():
            os.remove(path)
        if parent.exists() and len(os.listdir(parent)) == 0:
            os.rmdir(parent)

    return gdf


def preprocess_geo_background_source(
    background: None | bool | str | PathLike | gpd.GeoDataFrame,
    geometry_column: str,
) -> bool | Path:
    """
    Preprocesses the background input for geospatial analysis.

    Parameters:
        background: The input background, which can be:
            - None: Indicates no background data.
            - bool: A flag indicating whether background data is present.
            - str | PathLike: A file path to a shapefile (.shp).
            - GeoDataFrame: A GeoDataFrame containing background data.
        geometry_column: The name of the column containing geometry data.

    Returns:
        - bool: If `background` is None or a boolean.
        - GeoDataFrame: A GeoDataFrame with valid geometries.

    Raises:
        ValueError: If input lacks required columns or valid geometries.
        TypeError: If `background` is of an unsupported type.
    """
    if background is None or isinstance(background, bool):
        return False if background is None else background
    return preprocess_geo_source(background, geometry_column)


def preprocess_geo_source(
    source: str | PathLike | gpd.GeoDataFrame,
    geometry_column: str,
) -> Path:
    def validate_geometry(gdf):
        """
        Validates the GeoDataFrame for geometry data.

        Parameters:
            gdf: A GeoDataFrame object to validate.

        Returns:
            - None

        Raises:
            ValueError: If the geometry column column is valid.
        """
        if not (isinstance(gdf, gpd.GeoDataFrame) or geometry_column in gdf.columns):
            raise ValueError(
                f"The input must contain either a '{geometry_column}' column."
            )

    if isinstance(source, str):
        source = Path(source)
    if isinstance(source, Path):
        if source.suffix == ".shp":
            try:
                source_gdf = load_shapefile(source)
                if source_gdf.empty:
                    raise ValueError("The shapefile contains no data.")
                validate_geometry(source_gdf)
                return source
            except Exception as e:
                raise ValueError(f"Failed to load shapefile: {e}")
        else:
            raise ValueError("Source path must point to a shapefile (.shp).")

    elif isinstance(source, DataFrameLike):
        timestamp = f"{time.time()}"
        timestamp = timestamp[: timestamp.find(".")]
        source_path = (
            FTCNN_TMP_DIR / f"{TMP_FILE_PREFIX}preprocess_geo_source_{timestamp}.shp"
        )
        save_as_shp(source, source_path)
        return preprocess_geo_source(source_path, geometry_column)
    return Path(source)


def postprocess_geo_source(
    source: Path,
) -> None:
    if source.stem.startswith(TMP_FILE_PREFIX):
        source.unlink()


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
