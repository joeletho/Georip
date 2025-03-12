from typing import Union

from shapely import MultiPolygon, Polygon

PolygonLike = Union[Polygon, MultiPolygon]


def stringify_points(points: list[tuple[int | float, int | float]]) -> str:
    """
    Converts a list of points into a space-separated string representation.

    Parameters:
        points (list[tuple[int | float, int | float]]): A list of (x, y) coordinates.

    Returns:
        str: A string where each point is represented as "x y" and points are space-separated.

    Example:
        Input: [(1, 2), (3, 4)]
        Output: "1 2 3 4"
    """
    return " ".join([f"{point[0]} {point[1]}" for point in points])


def stringify_bbox(bbox: list[int | float]) -> str:
    """
    Converts a bounding box into a space-separated string representation.

    Parameters:
        bbox (list[int | float]): A list representing a bounding box [min_x, min_y, max_x, max_y].

    Returns:
        str: A space-separated string representation of the bounding box.

    Example:
        Input: [0, 0, 10, 10]
        Output: "0 0 10 10"
    """
    return f"{' '.join([str(x) for x in bbox])}"


def parse_points_list_str(s: str) -> list[tuple[float, float]]:
    """
    Parses a string representation of a list of points into a list of (x, y) tuples.

    Parameters:
        s (str): A string containing points in the format "(x1, y1)(x2, y2)...".

    Returns:
        list[tuple[float, float]]: A list of points as (x, y) tuples.

    Example:
        Input: "(1.0, 2.0)(3.0, 4.0)"
        Output: [(1.0, 2.0), (3.0, 4.0)]
    """
    points = []
    i = 0
    while i < len(s):
        if s[i] == "(":
            i += 1
            stop = s.index(")", i)
            point = s[i:stop].split(",")
            points.append((float(point[0]), float(point[1])))
            i = stop
        else:
            i += 1
    return points


def clip_points(points, shape):
    """
    Clips a list of points to ensure they fall within the bounds of a given shape.

    Parameters:
        points (list[tuple[int | float, int | float]]): A list of (x, y) coordinates.
        shape (tuple[int, int]): A tuple representing the width and height of the bounding shape (width, height).

    Returns:
        list[tuple[int | float, int | float]]: A list of clipped points, ensuring all (x, y) values fall within the range
                                               [0, width - 1] and [0, height - 1].

    Example:
        Input: [(5, 5), (10, -1)], shape=(10, 10)
        Output: [(5, 5), (9, 0)]
    """
    width = shape[0]
    height = shape[1]
    clipped = []
    for x, y in points:
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        clipped.append((x, y))
    return clipped


def normalize_point(
    x: int | float,
    y: int | float,
    width: int | float,
    height: int | float,
    *,
    xoffset: int | float | None = None,
    yoffset: int | float | None = None,
    include_dims: bool = False,
) -> tuple[float, float, float, float] | tuple[float, float]:
    """
    Normalizes a point's coordinates within a given width and height, applying optional offsets if provided.

    Parameters:
        x (int | float): The x-coordinate of the point.
        y (int | float): The y-coordinate of the point.
        width (int | float): The width of the bounding area for normalization.
        height (int | float): The height of the bounding area for normalization.
        xoffset (int | float | None): Optional x-coordinate offset.
        yoffset (int | float | None): Optional y-coordinate offset.

    Returns:
        tuple: A tuple containing the normalized x and y values. If offsets are provided,
               the normalized xoffset and yoffset are also included.

    Explanation:
        - `dw` and `dh` are scale factors computed from the width and height.
        - If offsets are provided, they are added to `x` and `y` before scaling and are also normalized.
        - If offsets are not provided, only `x` and `y` are normalized.
        - The result tuple only includes the normalized values for offsets if they were provided.
    """
    dw = 1 / float(width)  # Scale factor for x
    dh = 1 / float(height)  # Scale factor for y
    x = float(x)
    y = float(y)

    # Normalize coordinates and apply offsets if provided
    if xoffset is not None and yoffset is not None:
        xoffset = float(xoffset)
        yoffset = float(yoffset)
        x = (x + xoffset) * dw
        y = (y + yoffset) * dh
        xoffset *= dw
        yoffset *= dh
        result = round(x, 6), round(y, 6), round(xoffset, 6), round(yoffset, 6)
    else:
        x *= dw
        y *= dh
        result = round(x, 6), round(y, 6), round(dw, 6), round(dh, 6)
    return result if include_dims else result[:2]
