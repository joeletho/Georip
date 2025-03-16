from os import PathLike
from typing import Callable

import geopandas as gpd
import pandas as pd
from shapely import MultiPolygon, unary_union

from georip.geometry import PolygonLike


def normalize_fields(fields):
    """
    Normalizes the input fields into a consistent dictionary format, handling both
    strings and dictionaries within a list.

    Parameters:
        fields (list[Union[str, dict[str, str]]] | dict[str, str] | None): Fields to normalize.

    Returns:
        dict: A dictionary mapping old field names to new field names.
    """
    field_map = {}

    if fields:
        if isinstance(fields, dict):
            # If fields is already a dictionary, use it as-is.
            field_map = fields
        elif isinstance(fields, list):
            # Iterate over the list and handle strings and dictionaries separately.
            for item in fields:
                if isinstance(item, str):
                    # Preserve field as-is.
                    field_map[item] = item
                elif isinstance(item, dict):
                    # Add mappings from old_name to new_name.
                    field_map.update(item)
                else:
                    raise ValueError(f"Unsupported list item type: {type(item)}")
        else:
            raise ValueError(f"Unsupported fields type: {type(fields)}")

    return field_map


def extract_fields(data_row, field_map):
    """
    Extracts and optionally maps fields from a data row based on the specified configuration.

    Parameters:
        data_row (pd.Series): A row from a DataFrame or GeoDataFrame.
        fields (list[Union[str, dict[str, str]]] | dict[str, str] | None): Fields to extract. Can be:
            - A dictionary mapping old field names to new field names.
            - A list of field names or dictionaries for renaming.
            - None, in which case no fields are extracted.

    Returns:
        dict: A dictionary containing the extracted fields.
    """
    return {
        new_name: data_row.get(old_name, None)
        for old_name, new_name in field_map.items()
    }


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
