import sys
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import rasterio
import shapely
from PIL import Image
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from georip.geometry.polygons import create_tile_polygon
from georip.geoprocessing.utils import (
    stem_contains_region_and_years,
    stem_contains_years,
)
from georip.io import collect_files_with_suffix
from georip.raster import create_window
from georip.raster.utils import get_rows_cols_min_max_bounds
from georip.utils import StrPathLike
from georip.utils.pandas import extract_fields, normalize_fields


def update_metadata_region_name_and_years(
    metadata,
    region_column,
    start_year_column,
    end_year_column,
    region_name,
    start_year,
    end_year,
):
    for region in region_column:
        if not metadata.get(region):
            metadata[region] = region_name

    resolved_start_year_column = metadata.get(start_year_column, "start_year")
    resolved_end_year_column = metadata.get(end_year_column, "end_year")
    if not metadata.get(resolved_start_year_column):
        metadata[resolved_start_year_column] = start_year
    if not metadata.get(resolved_end_year_column):
        metadata[resolved_end_year_column] = end_year


def map_metadata(
    gdf_src: gpd.GeoDataFrame,
    images_dir: StrPathLike,
    region_column: str | list[str],
    start_year_column: str,
    end_year_column: str,
    preserve_fields: (
        Union[list[Union[str, dict[str, str]]], dict[str, str]] | None
    ) = None,
) -> gpd.GeoDataFrame:
    """
    Maps metadata for images referenced in a GeoDataFrame, with flexible field preservation and renaming.
    Ensures that specified columns exist in the source DataFrame before preservation or renaming.

    Parameters:
        gdf_src (gpd.GeoDataFrame): Source GeoDataFrame containing metadata.
        img_dir (StrPathLike): Directory containing image files.
        parse_filename (Callable): Function to derive filenames from GeoDataFrame rows.
        preserve_fields (Union[List[Union[str, Dict[str, str]]], Dict[str, str]], optional):
            Specifies fields to preserve from the original GeoDataFrame.
            Can be:
                - A list of strings: Columns to preserve as-is.
                - A list of dictionaries: Specifies renaming with `{new_name: old_name}`.
                - A dictionary: Specifies renaming in `{new_name: old_name}` format.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with image metadata and preserved/renamed fields.

    Raises:
        KeyError: If any column to preserve does not exist in the original DataFrame.
    """
    images_dir = Path(images_dir).resolve()
    rows = []
    geometry = []
    gdf = gdf_src.copy()

    image_paths = collect_files_with_suffix(
        [".tif", ".tiff", ".jpg", ".jpeg", ".png"], images_dir, recurse=True
    )
    if not len(image_paths):
        raise FileNotFoundError(
            f"Could not find images of type '.tif' or '.tiff' in {images_dir}"
        )

    if not isinstance(region_column, list):
        region_column = [region_column]

    image_paths.sort(key=lambda p: p.stem)
    gdf = gdf.sort_values(
        by=[*region_column, start_year_column, end_year_column]
    ).reset_index(drop=True)

    field_map = dict()
    if preserve_fields:
        field_map = normalize_fields(preserve_fields)

    for filepath in image_paths:
        path_stem = filepath.stem
        metadata = {
            "filename": filepath.name,
            "dirpath": str(filepath.parent),
            "width": None,
            "height": None,
            "bbox": None,
        }

        region_name = None
        start_year = None
        end_year = None

        row_indices = None
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
                    years_mask = (gdf[start_year_column] == start_year) & (
                        gdf[end_year_column] == end_year
                    )

                    region_mask = pd.Series([False] * len(gdf))
                    for col in region_column:
                        if (gdf[col] == region_name).any():
                            region_mask |= gdf[col] == region_name
                            break
                    row_indices = gdf[years_mask & region_mask].index.tolist()
                    break

            if row_indices is not None:
                break

        if row_indices is None or not len(row_indices):
            continue

        if filepath.suffix in [".tiff", ".tif"]:
            with rasterio.open(filepath, crs=gdf.crs) as img:
                width, height = img.width, img.height
        else:
            with Image.open(filepath) as img:
                width, height = img.size

        metadata["width"] = width
        metadata["height"] = height

        for index in row_indices:
            row = gdf.iloc[index]

            current_metadata = metadata.copy()

            if preserve_fields:
                current_metadata.update(extract_fields(row, field_map))

            update_metadata_region_name_and_years(
                current_metadata,
                region_column,
                start_year_column,
                end_year_column,
                region_name,
                start_year,
                end_year,
            )

            current_metadata["bbox"] = row.get("bbox")
            matching_geometry = row.get("geometry")

            if matching_geometry is None:
                continue
            if row.get("bbox") is None:
                current_metadata["bbox"] = shapely.box(*matching_geometry.bounds)

            rows.append(current_metadata)
            geometry.append(matching_geometry)

        gdf = gdf.drop(row_indices).reset_index(drop=True)

    if not len(rows):
        raise ValueError("No metadata found for mapping")

    return (
        gpd.GeoDataFrame(
            gpd.GeoDataFrame.from_dict(
                rows,
                geometry=geometry,
                crs=gdf_src.crs,
            )
            .explode()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        .sort_values(by=[*region_column, "start_year", "end_year"])
        .reset_index(drop=True)
    )


def map_geometry_to_geotiffs(
    gdf: gpd.GeoDataFrame,
    image_paths: list[StrPathLike],
    preserve_fields: list[str | dict[str, str]] | None = None,
) -> gpd.GeoDataFrame:
    """
    Maps geometries in a GeoDataFrame to corresponding GeoTIFF files based on spatial intersections.

    Parameters:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing geometries to map.
        images_dir (StrPathLike): The directory containing GeoTIFF files to map geometries to.
        recurse (bool): Whether to search for GeoTIFFs recursively within the directory. Defaults to True.
        **kwargs: Additional arguments, including `preserve_fields` for retaining specific fields.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing metadata for each GeoTIFF, including the intersecting geometries.
    """
    rows = []
    geometry = []

    image_paths = [Path(path) for path in image_paths]
    field_map = dict()
    if preserve_fields:
        field_map = normalize_fields(preserve_fields)

    # Iterate over each GeoTIFF file and map intersecting geometries
    for path in tqdm(
        image_paths,
        desc="Mapping geometry to GeoTIFFs",
        leave=False,
    ):
        with rasterio.open(path, crs=gdf.crs) as src:
            # Create a window that represents the full extent of the GeoTIFF
            (rmin, _), (cmin, _) = get_rows_cols_min_max_bounds(src)
            tile_window = create_window(cmin, rmin, src.width, src.height)

            # Create a polygon representing the bounds of the GeoTIFF
            tile_polygon = create_tile_polygon(src, tile_window)

            # Find polygons in the GeoDataFrame that intersect with the GeoTIFF polygon
            intersecting_polygons = gdf.loc[gdf.intersects(tile_polygon)]

            row = {
                "filename": path.name,
                "dirpath": str(path.parent),
                "width": src.width,
                "height": src.height,
            }

            if not intersecting_polygons.empty:
                for _, polygon_row in intersecting_polygons.iterrows():
                    row = {**row, **extract_fields(polygon_row, field_map)}
                    geometry.append(polygon_row["geometry"].intersection(tile_polygon))
                    rows.append(row)
            else:
                geometry.append(Polygon())
                rows.append(row)

    # Create the resulting GeoDataFrame
    result_gdf = gpd.GeoDataFrame.from_dict(rows, geometry=geometry, crs=gdf.crs)

    # Explode multi-part geometries and drop duplicates
    return gpd.GeoDataFrame(result_gdf.explode(ignore_index=True).drop_duplicates())


def map_geometries_by_year_span(
    gdf: gpd.GeoDataFrame,
    images_dir: StrPathLike,
    start_year_col: str,
    end_year_col: str,
    preserve_fields: list[str | dict[str, str]] | None = None,
):
    """
    Maps geometries to GeoTIFFs based on unique start and end year pairs.

    This function groups rows in the input GeoDataFrame by unique pairs of start and end years.
    For each year pair, it extracts the corresponding rows, applies a user-defined geometry
    mapping function to generate GeoTIFFs, and reinserts the year columns into the resulting
    mapped GeoDataFrame. The processed GeoDataFrames are returned as a list.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame containing geometries and year columns.
        start_year_col (str): The name of the column representing the start year.
        end_year_col (str): The name of the column representing the end year.
        map_geometry_to_geotiffs (function): A user-defined function that processes a
                                             GeoDataFrame and generates GeoTIFFs.

    Returns:
        list[GeoDataFrame]: A list of GeoDataFrames containing the mapped geometries,
                            with start and end year columns added.
    """
    gdf = gdf.copy()
    gdf[start_year_col] = gdf[start_year_col].apply(lambda year: int(year))
    gdf[end_year_col] = gdf[end_year_col].apply(lambda year: int(year))

    # Identify unique year pairs and sort them
    year_pairs = gdf[[start_year_col, end_year_col]].drop_duplicates()
    year_pairs = year_pairs.sort_values(by=[start_year_col, end_year_col])
    start_years = [int(year) for year in year_pairs[start_year_col].tolist()]
    end_years = [int(year) for year in year_pairs[end_year_col].tolist()]

    mapped_gdfs = []
    images = collect_files_with_suffix([".tif", ".tiff"], images_dir, recurse=True)

    for start_year, end_year in zip(start_years, end_years):
        target_images = [
            path
            for path in images
            if stem_contains_years(path.stem, start_year, end_year)
        ]
        if not len(target_images):
            print(
                f"No images found for years {start_year} to {end_year}", file=sys.stderr
            )
            continue

        # Filter rows that match the current year pair
        target_years = gpd.GeoDataFrame(
            gdf[(gdf[start_year_col] == start_year) & (gdf[end_year_col] == end_year)],
            crs=gdf.crs,
        )
        # Apply the mapping function to the filtered rows
        gdf_mapped = map_geometry_to_geotiffs(
            target_years,
            target_images,
            preserve_fields=preserve_fields,
        )
        if not len(([True for col in gdf_mapped.columns if col == start_year_col])):
            # Add the year columns back into the resulting GeoDataFrame
            gdf_mapped.insert(0, start_year_col, int(start_year))
            gdf_mapped.insert(1, end_year_col, int(end_year))
        else:
            gdf_mapped[start_year_col] = int(start_year)
            gdf_mapped[end_year_col] = int(end_year)

        mapped_gdfs.append(gdf_mapped)

    return mapped_gdfs
