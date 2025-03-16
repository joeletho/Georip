import rasterio
from shapely import Polygon
from shapely.errors import InvalidGeometryError

from georip.geometry import clip_points
from georip.geometry.polygons import normalize_polygon
from georip.utils import StrPathLike


def __append_points(src, points: list, out: list, to_type: str):
    """
    Helper function to translate a list of points between geospatial (XY) and pixel index coordinates.

    Parameters:
        src: rasterio.DatasetReader
            An open rasterio dataset used to determine coordinate transformations.
        points: list
            A list of points to be translated, where each point is a tuple of coordinates.
        out: list
            A list that will be updated with the translated points.
        to_type: str
            The target coordinate system: either "xy" for geospatial or "index" for pixel index.

    Raises:
        Exception: If an unknown `to_type` is provided.
    """
    for point in points:
        match (to_type):
            case "xy":
                translated = src.xy(point[1], point[0])
            case "index":
                translated = src.index(point[0], point[1])[::-1]
            case _:
                raise Exception(f"Unknown type '{to_type}")
        out.append(translated)


def __translate_polygon(source_path, polygon, to_type):
    """
    Helper function to translate the coordinates of a polygon between geospatial (XY)
    and pixel index systems using a raster file as the source of reference.

    Parameters:
        source_path: StrPathLike
            Path to the raster file used for coordinate transformations.
        polygon: Polygon
            The input polygon to translate.
        to_type: str
            The target coordinate system: either "xy" for geospatial or "index" for pixel index.

    Returns:
        Polygon: A polygon with coordinates translated to the target system.
    """
    translated = []
    with rasterio.open(source_path) as src:
        points = list(polygon.exterior.coords)
        __append_points(src, points, translated, to_type)
    return Polygon(clip_points(translated, (src.width, src.height)))


def translate_polygon_xy_to_index(
    source_path: StrPathLike,
    polygon: Polygon | str | list,
) -> Polygon:
    """
    Translates a polygon from geospatial (XY) coordinates to pixel index coordinates
    relative to the provided raster file.

    Parameters:
        source_path: StrPathLike
            Path to the GeoTIFF raster file used for coordinate transformations.
        polygon: Polygon | str | list
            The input polygon in geospatial (XY) coordinates. Can be:
                - A shapely Polygon object
                - A string representation of a polygon
                - A list of coordinate tuples

    Returns:
        Polygon: A polygon with pixel index coordinates.

    Raises:
        ValueError: If the polygon cannot be normalized or processed.
    """
    polygon = normalize_polygon(polygon)
    if not polygon.is_valid:
        raise InvalidGeometryError("Polygon does not have valid geometry")
    return __translate_polygon(source_path, polygon, "index")


def translate_polygon_index_to_xy(
    source_path: StrPathLike,
    polygon: Polygon | str | list,
) -> Polygon:
    """
    Translates a polygon from pixel index coordinates to geospatial (XY) coordinates
    relative to the provided raster file.

    Parameters:
        source_path: StrPathLike
            Path to the GeoTIFF raster file used for coordinate transformations.
        polygon: Polygon | str | list
            The input polygon in pixel index coordinates. Can be:
                - A shapely Polygon object
                - A string representation of a polygon
                - A list of coordinate tuples

    Returns:
        Polygon: A polygon with geospatial (XY) coordinates.

    Raises:
        ValueError: If the polygon cannot be normalized or processed.
    """
    polygon = normalize_polygon(polygon)
    if not polygon.is_valid:
        raise InvalidGeometryError("Polygon does not have valid geometry")
    return __translate_polygon(source_path, polygon, "xy")
