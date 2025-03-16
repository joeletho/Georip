import sys

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.io import DatasetWriter
from rasterio.windows import Window
from shapely import box, normalize, remove_repeated_points
from shapely.geometry import Polygon
from supervision.annotators.core import cv2

from georip.geometry import PolygonLike
from georip.utils.pandas import extract_fields, normalize_fields


def flatten_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Flattens MultiPolygons into individual Polygons and adds bounding boxes.

    Parameters:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with geometries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with flattened polygons and bounding boxes.
    """
    flattened_gdf = gdf.explode(index_parts=False)
    return flattened_gdf.reset_index(drop=True)


# def flatten_polygons(
#     gdf_src: gpd.GeoDataFrame,
#     geometry_column: str = "geometry",
#     group_by: str | list[str] | None = None,
#     preserve_fields: list[str | dict[str, str]] | None = None,
# ) -> gpd.GeoDataFrame:
#     """
#     Flattens MultiPolygons into individual Polygons and calculates bounding boxes for each geometry.
#     The function handles grouping by specified columns and returns a new GeoDataFrame where each row
#     corresponds to a single Polygon (from MultiPolygon) and its associated bounding boxes.
#
#     Parameters:
#         gdf_src (gpd.GeoDataFrame): The source GeoDataFrame containing geometries.
#         geometry_column (str, optional): The column name containing the geometries to be flattened (default is "geometry").
#         group_by (Union[str, list[str], None], optional): Specifies columns by which to group the data
#             before applying the flattening and bounding box calculation. Defaults to None (no grouping).
#
#     Returns:
#         gpd.GeoDataFrame: A new GeoDataFrame where MultiPolygon geometries are flattened into individual
#             Polygons, with bounding boxes included for each geometry.
#
#     Notes:
#         - For MultiPolygon geometries, each individual polygon is extracted and treated as a separate row.
#         - The bounding boxes are calculated for each polygon and added as a new field.
#     """
#     geometry = []
#     rows = []
#     field_map = dict()
#     if preserve_fields:
#         field_map = normalize_fields(preserve_fields)
#
#     for _, group in gdf_src.groupby(group_by, sort=False):
#         row = group.iloc[0].drop(geometry_column).to_dict()
#         polygon = shapely.unary_union(group.geometry)
#
#         if isinstance(polygon, shapely.MultiPolygon):
#             for poly in polygon.geoms:
#                 poly = remove_repeated_points(normalize(poly))
#                 if not poly.is_valid:
#                     continue
#                 if poly.has_z:
#                     poly = Polygon([(x, y) for x, y, _ in poly.exterior.coords])
#                 existing_fields = extract_fields(row, field_map)
#                 for bbox in get_polygon_bboxes(poly):
#                     row["bbox"] = bbox
#                     rows.append({**row, **existing_fields})
#                     geometry.append(poly)
#         else:
#             polygon = remove_repeated_points(normalize(polygon))
#             if not polygon.is_valid:
#                 continue
#             if polygon.has_z:
#                 polygon = Polygon([(x, y) for x, y, _ in polygon.exterior.coords])
#             existing_fields = extract_fields(row, field_map)
#             for bbox in get_polygon_bboxes(polygon):
#                 row["bbox"] = bbox
#                 rows.append({**row, **existing_fields})
#                 geometry.append(polygon)
#
#     return gpd.GeoDataFrame.from_dict(rows, geometry=geometry, crs=gdf_src.crs)
#


def get_polygon_points(
    polygon: PolygonLike,
) -> list[tuple[int | float]] | list[list[tuple[int | float, int | float]]]:
    """
    Extracts the coordinates of a polygon or multipolygon as a list of points.

    This function retrieves the coordinates of a polygon or multipolygon, returning a list of tuples
    representing the points for each polygon. For multipolygons, the function returns a list of lists
    of points for each individual polygon.

    Parameters:
        polygon (PolygonLike): The input geometry, which can be a Polygon or MultiPolygon.

    Returns:
        list[tuple[int | float]] | list[list[tuple[int | float, int | float]]]:
            A list of points representing the polygon's exterior coordinates.
            For MultiPolygons, a list of lists is returned, each containing the points of a separate polygon.

    Raises:
        ValueError: If the input geometry is not a Polygon or MultiPolygon.
    """
    match (polygon.geom_type):
        case "Polygon":
            points = [point for point in polygon.exterior.coords]
        case "MultiPolygon":
            points = [
                [point for point in poly.exterior.coords] for poly in polygon.geoms
            ]
        case _:
            raise ValueError("Unknown geometry type")
    return points


def get_polygon_bboxes(geom: PolygonLike) -> list[Polygon]:
    """
    Calculates bounding boxes for polygons or multipolygons.

    This function computes bounding boxes for polygons or multipolygons and returns them as
    list(s) of Polygon objects representing the bounding boxes of the input geometry.

    Parameters:
        geom (PolygonLike): The input geometry, which can be a Polygon or MultiPolygon.

    Returns:
        list[Polygon]: A list of Polygon objects representing the bounding boxes of the input geometry.

    Example:
        For a multipolygon, this will return a list of bounding boxes, one for each individual polygon.
    """
    boxes = []
    match (geom.geom_type):
        case "Polygon":
            boxes.append(normalize(shapely.box(*geom.bounds)))
        case "MultiPolygon":
            for geom in geom.geoms:
                boxes.append(normalize(shapely.box(*geom.bounds)))
        case _:
            print(f"Unknown geometry type '{geom.geom_type}'", file=sys.stderr)
    return boxes


def get_geom_polygons(geom: PolygonLike, *, flatten: bool = False) -> list[Polygon]:
    """
    Extracts polygons from a geometry, with an option to flatten multipolygons.

    This function processes a polygon or multipolygon geometry and returns a list of individual polygons.
    For multipolygons, it can either return each polygon separately or merge (flatten) overlapping polygons
    into a single polygon, depending on the `flatten` flag.

    Parameters:
        geom (PolygonLike): The input geometry, which can be a Polygon or MultiPolygon.
        flatten (bool): A flag to determine whether to flatten overlapping polygons into one (default: False).

    Returns:
        list[Polygon]: A list of polygons extracted from the input geometry, either as individual polygons
                        or as merged polygons if `flatten` is set to True.

    Raises:
        ValueError: If the input geometry is not a Polygon or MultiPolygon.
    """
    polygons = []

    match (geom.geom_type):
        case "Polygon":
            polygons.append(normalize(geom))
        case "MultiPolygon":
            if not flatten:
                polygons.extend([normalize(g) for g in list(geom.geoms)])
            else:
                flattened = []
                for polygon in [normalize(p) for p in list(geom.geoms)]:
                    if len(flattened) == 0:
                        flattened.append(polygon)
                        continue
                    found = False
                    for i, flat in enumerate(flattened):
                        union = shapely.coverage_union(flat, polygon)
                        for poly in union.geoms:
                            # There is no union
                            if poly.equals(flat) or poly.equals(polygon):
                                continue
                            # Replace this polygon with the union
                            flattened[i] = poly
                            found = True
                            break
                        if found:
                            break
                    if not found:
                        flattened.append(polygon)
                polygons.extend(flattened)

        case _:
            raise ValueError("Unknown geometry type")
    return polygons


"""
    This might be the issue which causes the label issues where geometry 
    does not align with the actual geom,also causing the labeled geom to 
    appear "clipped" overlayed on the image
"""


def create_tile_polygon(src: DatasetWriter, tile_window: Window) -> Polygon:
    """
    Creates a polygon representing the geographic bounds of a raster tile.

    The function uses the provided raster dataset and a tile window to calculate the coordinates of the
    tile’s four corners in the spatial reference of the raster. These coordinates are then used to create
    a polygon that outlines the tile’s bounding box.

    Parameters:
        src (rasterio.io.DatasetWriter): The raster dataset from which the tile's spatial reference is derived.
        tile_window (rasterio.windows.Window): The window defining the raster tile, including its size and position.

    Returns:
        Polygon: A Polygon object representing the geographic bounds of the raster tile.

    Example:
        Given a raster dataset and a tile window, the function returns a Polygon representing the tile’s bounds
        in the coordinate reference system of the raster.
    """
    tile_transform = rasterio.Affine(*src.window_transform(tile_window))
    width: int = tile_window.width
    height: int = tile_window.height

    tile_polygon = Polygon(
        [
            tile_transform * (0, 0),  # Top-left
            tile_transform * (width, 0),  # Top-right
            tile_transform * (width, height),  # Bottom-right
            tile_transform * (0, height),  # Bottom-left
            tile_transform * (0, 0),  # Close the polygon (back to top-left)
        ]
    )
    return tile_polygon


def parse_polygon_str(polygon_str: str):
    """
    Parses a polygon string into a list of coordinate tuples.

    The function extracts the coordinates from a well-formed polygon string, where each coordinate pair is
    separated by a comma and space, and coordinates are enclosed in parentheses. The string is stripped of
    any leading or trailing non-digit characters, then the coordinates are parsed and returned as a list of tuples.

    Parameters:
        polygon_str (str): A string representation of a polygon with coordinates, such as:
            "(x1 y1, x2 y2, ..., xn yn)".

    Returns:
        list[tuple[float, float]]: A list of coordinate tuples [(x1, y1), (x2, y2), ..., (xn, yn)].

    Example:
        Given the input: "(1 2, 3 4, 5 6)",
        The function will return: [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)].
    """
    size = len(polygon_str)
    start = 0
    end = size - 1
    while (
        start < end
        and start < size
        and end >= 0
        and not (polygon_str[start].isdigit() and polygon_str[end].isdigit())
    ):
        if not polygon_str[start].isdigit():
            start += 1
        if not polygon_str[end].isdigit():
            end -= 1

    if start < size and end >= 0:
        polygon_str = polygon_str[start : end + 1]
    parsed = []
    points = polygon_str.split(", ")
    for point in points:
        point = point.replace(" 0", "").replace("(", "").replace(")", "")
        x, y = point.split()
        parsed.append((float(x), float(y)))

    return parsed


def normalize_polygon(
    polygon: Polygon | str | list[tuple[int | float, int | float]] | list[int | float]
) -> Polygon:
    """
    Normalizes a polygon by converting various input formats into a simplified Polygon geometry.

    This function accepts multiple formats for a polygon input:
    - A `Polygon` object: Directly normalized and simplified.
    - A list of coordinate pairs: Interpreted as a sequence of points to form a Polygon.
    - A WKT string: Parsed into a Polygon and then normalized.

    The normalization involves:
    - Ensuring the geometry is represented as a `Polygon` object.
    - Simplifying the polygon to reduce vertices while preserving topology.

    Parameters:
        polygon (Polygon | str | list[tuple[int | float, int | float]] | list[int | float]):
            The input polygon in one of the following formats:
            - A `Polygon` object.
            - A list of coordinate pairs (e.g., [(x1, y1), (x2, y2), ...]).
            - A WKT string representing the polygon.

    Returns:
        Polygon: A normalized `Polygon` object that has been simplified and processed.

    Raises:
        ValueError: If the input cannot be interpreted as a valid polygon.
    """

    if not isinstance(polygon, Polygon):
        if isinstance(polygon, list):
            if not isinstance(polygon[0], tuple):
                polygon = [
                    (polygon[i], polygon[i + 1]) for i in range(len(polygon) - 1)
                ]
        elif isinstance(polygon, str):
            polygon = parse_polygon_str(polygon)

    return remove_repeated_points(normalize(polygon), tolerance=0)


def mask_to_polygon(mask):
    """
    Converts a binary mask to polygons.

    Parameters:
        mask: np.ndarray
            Binary mask.

    Returns:
        List[Polygon]: List of polygons derived from the mask.
    """
    # Assuming mask is binary, extract polygons from mask
    mask = (mask * 255).astype(np.uint8).squeeze()
    # print(type(mask), mask.shape, mask.dtype, mask.min(), mask.max())
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = [Polygon(c.reshape(-1, 2)) for c in contours if len(c) >= 3]
    return polygons


def is_sparse_polygon(
    polygon: Polygon,
    *,
    min_area: int | float = 1,
    max_aspect_ratio: int | float = 10,
) -> bool:
    """
    Determines if a geometry is sparse (low area, long, and narrow).

    Args:
        polygon (Polygon): The input geometry (Polygon).
        min_area (float): Minimum area threshold before scaling.
        max_aspect_ratio (float): Maximum aspect ratio for considering the shape sparse.

    Returns:
        bool: True if the geometry is sparse, False otherwise.
    """
    if not isinstance(polygon, Polygon):
        raise ValueError("Input geometry must be a Polygon")

    xmin, ymin, xmax, ymax = polygon.bounds
    width = xmax - xmin
    height = ymax - ymin

    if width == 0 or height == 0:
        return True

    if polygon.is_empty:
        return True

    if polygon.has_z:
        polygon = Polygon(
            [(x / width, y / height) for x, y, _ in polygon.exterior.coords]
        )
    else:
        polygon = Polygon([(x / width, y / height) for x, y in polygon.exterior.coords])

    polygon = remove_repeated_points(normalize(polygon), tolerance=0)
    if not polygon.is_valid:
        return True

    scaling_factor = width * height
    scaled_min_area = min_area / scaling_factor if scaling_factor > 0 else float("inf")

    if polygon.area < scaled_min_area:
        return True

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > max_aspect_ratio:
        return True

    return False
