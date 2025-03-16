import sys
import time
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd
from geopandas.geoseries import shapely

import georip.io as io
from georip.geometry.polygons import is_sparse_polygon
from georip.geoprocessing import DataFrameLike
from georip.geoprocessing.utils import (
    gdf_intersects_region_year_geometry,
    update_region_bbox,
)
from georip.io.geoprocessing import load_shapefile
from georip.utils import GEORIP_TMP_DIR, StrPathLike

TMP_FILE_PREFIX = "tmp__"


def init_dataset_filepaths(
    *,
    source_shp: StrPathLike,
    source_images_dir: StrPathLike,
    output_dir: StrPathLike,
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
        io.clear_directory(output_dir)
    elif not output_dir.exists():
        output_dir.mkdir(parents=True)

    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=exist_ok)
    if save_shp or save_gpkg:
        shp_dir.mkdir(parents=True, exist_ok=exist_ok)

    tiles_dir: Path = output_dir / "images" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=exist_ok)
    return {
        "shapefile": Path(source_shp),
        "output_dir": Path(output_dir),
        "image_dir_src": Path(source_images_dir),
        "tiles_dir": Path(tiles_dir),
        "meta_dir": Path(meta_dir),
        "csv_dir": Path(csv_dir),
        "shp_dir": Path(shp_dir),
    }


def remove_unused_tiles(
    gdf: gpd.GeoDataFrame,
    geom_column: str,
    image_directory_column: str,
    image_filename_column: str,
) -> gpd.GeoDataFrame:
    """
    Removes invalid geometries and associated files, and cleans up empty directories.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing geometry and image paths.
        geom_col (str): The column name in `gdf` that contains geometries.
        image_directory_column (str): The column name containing the directory paths to images.
        image_filename_column (str): The column name containing the image filenames.

    Returns:
        GeoDataFrame: The updated GeoDataFrame with valid geometries and cleaned-up paths.
    """
    gdf = gdf.explode(column=geom_column, ignore_index=True)

    unique_paths = gdf[
        [image_directory_column, image_filename_column]
    ].drop_duplicates()

    for _, row in unique_paths.iterrows():
        directory = row[image_directory_column]
        filename = row[image_filename_column]
        full_path = Path(directory) / filename

        path_mask = (gdf[image_directory_column] == directory) & (
            gdf[image_filename_column] == filename
        )
        geometries = gdf.loc[path_mask, geom_column]

        sparse_mask = geometries.apply(is_sparse_polygon)
        gdf = gdf.drop(gdf.loc[path_mask & sparse_mask].index).reset_index(drop=True)

        if gdf.loc[path_mask].empty:
            if full_path.exists():
                try:
                    full_path.unlink()
                except OSError as e:
                    print(f"Error removing file {full_path}: {e}")

                parent_dir = full_path.parent
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    try:
                        parent_dir.rmdir()
                    except OSError as e:
                        print(f"Error removing directory {parent_dir}: {e}")

    return gdf


def preprocess_geo_background_source(
    background: None | bool | StrPathLike | gpd.GeoDataFrame,
    geometry_column: str,
) -> bool | Path:
    """
    Preprocesses the background input for geospatial analysis.

    Parameters:
        background: The input background, which can be:
            - None: Indicates no background data.
            - bool: A flag indicating whether background data is present.
            - StrPathLike: A file path to a shapefile (.shp).
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
    source: StrPathLike | gpd.GeoDataFrame,
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
            GEORIP_TMP_DIR / f"{TMP_FILE_PREFIX}preprocess_geo_source_{timestamp}.shp"
        )
        io.save_as_shp(source, source_path)
        return preprocess_geo_source(source_path, geometry_column)
    return Path(source)


def merge_source_and_background(config):
    filepaths = init_dataset_filepaths(
        source_shp=config["shapefile"],
        source_images_dir=config["image_dir_src"],
        output_dir=config["output_dir"],
        exist_ok=config["exist_ok"],
        save_csv=config["save_csv"],
        save_shp=config["save_shp"],
        save_gpkg=config["save_gpkg"],
        clean_dest=config["clear_output_dir"],
    )
    shapefile = filepaths["shapefile"]
    image_dir = filepaths["image_dir_src"]
    shp_dir = filepaths["shp_dir"]
    csv_dir = filepaths["csv_dir"]

    # Declare the path of the base files used for this dataset
    BASE_FILEPATH = Path(f"base_{shapefile.stem}")

    # Load the source shapefile
    gdf_source = io.load_shapefile(shapefile)

    # Fix a typo in the column name
    if gdf_source.get("gee_eregio") is not None:
        gdf_source.rename(columns={"gee_eregio": "gee_region"}, inplace=True)
        io.save_as_shp(gdf_source, shapefile, exist_ok=True)
    gdf_source.sort_values(by=[config["region_column"], config["class_column"]])

    gdf_source = gdf_source.drop(
        index=gdf_source.loc[
            ~gdf_source[config["class_column"]].isin(config["class_names"])
        ].index
    )
    gdf_source = gdf_source.sort_values(
        by=[config["region_column"], config["class_column"]]
    ).reset_index(drop=True)

    source_shp_path = shp_dir / BASE_FILEPATH.with_suffix(".shp")
    io.save_as_csv(
        gdf_source,
        csv_dir / BASE_FILEPATH.with_suffix(".csv"),
        exist_ok=True,
    )
    io.save_as_shp(gdf_source, source_shp_path, exist_ok=True)

    # Handle background first to be added to the source
    gdf_background = io.load_shapefile(config["background_shapefile"])

    # Fix a typo in the column name
    if gdf_background.get("gee_eregio") is not None:
        gdf_background.rename(columns={"gee_eregio": "gee_region"}, inplace=True)
        io.save_as_shp(gdf_background, config["background_shapefile"], exist_ok=True)

    if not gdf_background.crs:
        gdf_background = gdf_background.set_crs(gdf_source.crs)
    else:
        gdf_background = gdf_background.to_crs(gdf_source.crs)

    n_sample = len(gdf_source) * 5
    if gdf_background is not None:
        n_sample = min(len(gdf_background), n_sample)
    gdf_background = gdf_background.sample(
        n=n_sample, random_state=config["background_seed"]
    )
    gdf_background = gdf_background.reindex(
        columns=gdf_source.columns, fill_value="None"
    )
    gdf_background = update_region_bbox(
        gdf_background,
        config["region_column"],
        io.collect_files_with_suffix(".tiff", image_dir, recurse=True),
    )
    gdf_background[config["class_column"]] = "Background"

    background_save_name = Path(f"background_{BASE_FILEPATH}")
    background_shp_path = shp_dir / background_save_name.with_suffix(".shp")
    io.save_as_csv(
        gdf_background,
        csv_dir / background_save_name.with_suffix(".csv"),
        exist_ok=True,
    )
    io.save_as_shp(gdf_background, background_shp_path, exist_ok=True)
    # Merge the source and background dataframes
    source_merged = (
        gpd.GeoDataFrame(
            pd.concat([gdf_source, gdf_background], ignore_index=True),
            crs=gdf_source.crs,
        )
        .sort_values(
            by=[
                config["year_start_column"],
                config["class_column"],
                config["region_column"],
            ]
        )
        .reset_index(drop=True)
    )

    source_merged = source_merged.drop(
        index=source_merged[
            source_merged[config["year_start_column"]].astype(int) == 0
        ].index
    ).reset_index(drop=True)

    source_merged_save_name = Path(f"{BASE_FILEPATH}_merged")
    source_merged_shp_path = shp_dir / source_merged_save_name.with_suffix(".shp")
    io.save_as_csv(
        source_merged,
        csv_dir / source_merged_save_name.with_suffix(".csv"),
        exist_ok=True,
    )
    io.save_as_shp(source_merged, source_merged_shp_path, exist_ok=True)

    return source_merged_shp_path


def postprocess_geo_source(
    source: Path,
) -> None:
    if source.stem.startswith(TMP_FILE_PREFIX):
        source.unlink()


def _filter_geometry_caller(
    filepath, geometry, *, gdf, region_column, start_year_column, end_year_column
):
    return gdf_intersects_region_year_geometry(
        gdf=gdf,
        filepath=filepath,
        geometry=geometry,
        region_column=region_column,
        start_year_column=start_year_column,
        end_year_column=end_year_column,
    )


def _encode_classes(
    row: gpd.GeoSeries, geom_col: str, class_col: str, class_names: str | list
):
    """
    Encodes class labels based on the given class column and geometry validity.

    Parameters:
        row (GeoSeries): A single row of a GeoDataFrame containing geometry and class information.
        geom_col (str): The column name containing geometries.
        class_col (str): The column name containing class labels.
        class_names (str | list): A string or list of class names to match against.

    Returns:
        tuple[int, str]: A tuple containing the class ID and class name.

    Explanation:
        - If `class_names` is a string, it is converted to a list for consistency.
        - The function initializes the default classification as "Background" with an ID of -1.
        - If the geometry is missing, empty, or invalid, the function logs the issue and classifies the row as "Background."
        - Otherwise, it retrieves the class name from the specified column and assigns an ID based on its index in `class_names`.
    """
    if not isinstance(class_names, list):
        class_names = [class_names]
    class_name = "Background"
    class_id = -1

    geom = row.get(geom_col)
    if geom is None or geom.is_empty or not geom.is_valid:
        if geom is not None:
            print(
                "encode_classes: geom is invalid:",
                shapely.is_valid_reason(geom),
                file=sys.stderr,
            )
        else:
            print("encode_classes: Geometry is None", file=sys.stderr)
        print(
            "Row does not meet minimum criteria -- Classifying row as 'Background':",
            row,
            file=sys.stderr,
        )
    else:
        class_name = str(row.get(class_col))
        for i, name in enumerate(class_names):
            if class_name == name:
                class_id = i
    return (class_id, class_name)


def encode_classes(
    df: DataFrameLike, encoder: Callable | None = None, **encoder_kwargs
) -> DataFrameLike:
    """
    Adds encoded class information to a DataFrame.

    Parameters:
        df (DataFrameLike): The input DataFrame containing data to be encoded.
        encoder (Callable or None): A function that encodes a row into class ID and class name.
                            Defaults to `encode_default_classes`.
        encoder_kwargs (dict): Additional keyword arguments for default encoder function. Ignored if `encoder` is not None.

    Returns:
        DataFrameLike: A copy of the DataFrame with added "class_id" and "class_name" columns.
    """
    columns = {"class_id": [], "class_name": []}
    caller = (
        encoder
        if encoder is not None
        else lambda r: _encode_classes(r, **encoder_kwargs)
    )
    for _, row in df.iterrows():
        id, name = caller(row)
        columns["class_id"].append(id)
        columns["class_name"].append(name)
    df_encoded = df.copy()
    df_encoded.insert(0, "class_id", columns["class_id"])
    df_encoded.insert(1, "class_name", columns["class_name"])
    return df_encoded
