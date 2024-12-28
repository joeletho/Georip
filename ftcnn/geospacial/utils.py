from os import PathLike
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd
from shapely import MultiPolygon, Polygon, unary_union

from ftcnn.geometry import PolygonLike
from ftcnn.geometry.polygons import get_polygon_points
from ftcnn.geospacial import DataFrameLike
from ftcnn.geospacial.conversion import translate_polygon_xy_to_index


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


def parse_filename(series: pd.Series) -> str:
    """
    Constructs a filename based on specific fields in a pandas Series.

    Parameters:
        series (pd.Series): A pandas Series containing the fields "Subregion", "StartYear", and "EndYear".

    Returns:
        str: A constructed filename string in the format:
             "[Subregion]_Expanded_[StartYear]to[EndYear]_NDVI_Difference.tif".
    """
    subregion = str(series["Subregion"])
    startyear = str(series["StartYear"])
    endyear = str(series["EndYear"])

    years_part = "to".join([startyear, endyear])
    end_part = "NDVI_Difference.tif"

    filename = subregion
    last = filename[-1]
    if last.isdigit():
        filename += "_"
    elif last == "E":
        filename = "_".join([filename[:-1], "Expanded", ""])
    start_part = filename + years_part
    return "_".join([start_part, end_part])


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


def get_geometry(
    df: gpd.GeoDataFrame,
    *,
    geom_key: str = "geometry",
    parse_key: Callable | None = None,
) -> list[PolygonLike]:
    """
    Extracts geometries from a GeoDataFrame.

    Parameters:
        df (gpd.GeoDataFrame): The input GeoDataFrame containing geometry data.
        geom_key (str, optional): The column name containing the geometries. Defaults to "geometry".
        parse_key (Callable, optional): A callable to parse a row for geometry. If None, uses the `geom_key`.

    Returns:
        list[PolygonLike]: A list of geometries extracted from the GeoDataFrame.
    """
    geoms = []
    for _, row in df.iterrows():
        if parse_key is not None:
            geom = parse_key(row)
            if geom is not None:
                geoms.append(geom)
        else:
            geoms.append(row[geom_key])
    return geoms


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


def merge_overlapping_geometries(
    source_shp: PathLike, distance: int = 8
) -> gpd.GeoDataFrame:
    """
    Merges overlapping geometries in a shapefile by buffering and unionizing them.
    Attributes are combined for merged geometries, and smaller polygons inside larger ones are removed.

    Parameters:
        source_shp (PathLike): Path to the input shapefile.
        distance (int): Buffer distance to expand geometries for intersection checks.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the merged geometries with updated attributes.
    """
    gdf = gpd.read_file(
        source_shp
    )  # read from shape file and create geopandas dataframe

    modified_gdf = gpd.GeoDataFrame(
        columns=gdf.columns, crs=gdf.crs
    )  # create a new dataframe for output

    rows = []

    # iterate checking each polygon with everyother polygon
    for j in range(len(gdf) - 1):
        ply1 = gdf.iloc[
            j
        ].geometry  # Add the polygon to the dataframe in case it doesn't get unionized
        ply1_attributes = gdf.iloc[j].drop("geometry").to_dict()  # Extract attributes
        rows.append(
            {**ply1_attributes, "geometry": ply1}
        )  # Add the polygon and attributes

        for i in range(j + 1, len(gdf)):
            ply2 = gdf.iloc[i].geometry
            ply2_attributes = (
                gdf.iloc[i].drop("geometry").to_dict()
            )  # Extract attributes

            ply1_buf = ply1.buffer(distance)
            ply2_buf = ply2.buffer(distance)

            if ply1_buf.intersects(
                ply2_buf
            ):  # If the dilated polygons intersect, unionize
                result = unary_union([ply2_buf, ply1_buf])
                minx, miny, maxx, maxy = result.bounds

                bbox_x = minx
                bbox_y = miny
                bbox_w = maxx - minx
                bbox_h = maxy - miny

                new_row = {**ply1_attributes, **ply2_attributes}
                new_row.update(
                    {
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "geometry": result,
                    }
                )
                rows.append(new_row)

    # Create a new GeoDataFrame with the new polygons
    new_rows_gdf = gpd.GeoDataFrame(rows, crs=gdf.crs)
    modified_gdf = pd.concat([modified_gdf, new_rows_gdf], ignore_index=True)

    # Clean up smaller polygons that may still be in larger ones
    unioned_geometry = unary_union(modified_gdf.geometry)

    # If the result is a MultiPolygon, split it back into individual polygons
    if isinstance(unioned_geometry, MultiPolygon):
        unioned_rows = []
        for geom in unioned_geometry.geoms:
            # Find overlapping polygons to retain attributes
            overlapping_rows = modified_gdf[modified_gdf.geometry.intersects(geom)]
            if not overlapping_rows.empty:
                representative_row = overlapping_rows.iloc[
                    0
                ].to_dict()  # Take attributes of the first match
                representative_row["geometry"] = geom
                unioned_rows.append(representative_row)
    else:
        representative_row = modified_gdf.iloc[
            0
        ].to_dict()  # Take attributes of the first row
        representative_row["geometry"] = unioned_geometry
        unioned_rows = [representative_row]

    # Create a new GeoDataFrame from the unioned geometries
    return gpd.GeoDataFrame(unioned_rows, columns=["geometry"], crs=modified_gdf.crs)
