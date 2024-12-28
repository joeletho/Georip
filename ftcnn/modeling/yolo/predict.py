from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from geopandas import gpd
from osgeo.gdal import sys
from rasterio import rasterio
from shapely import MultiPolygon, Polygon, normalize, unary_union
from tqdm.auto import tqdm, trange

from ftcnn.geometry.polygons import mask_to_polygon
from ftcnn.geospacial.utils import parse_subregion_and_years_from_path
from ftcnn.modeling.yolo.utils import get_result_stats
from ftcnn.raster.tools import tile_raster_and_convert_to_png
from ftcnn.utils import NUM_CPU


def predict_on_image_stream(model, *, images, conf=0.6, **kwargs):
    """
    Streams predictions for a batch of images using a model.

    Parameters:
        model: Model
            The YOLO model instance.
        images: List[np.ndarray]
            List of images to predict on.
        conf: float, optional
            Confidence threshold for predictions. Default is 0.6.
        kwargs: dict
            Additional arguments for the model's predict function.

    Yields:
        Tuple[ResultStats, Path]: Detection results and associated image paths.

    Raises:
        Exception: If prediction fails for any image batch.
    """
    batch_size = kwargs.get("batch_size")
    if batch_size is None:
        num_workers = kwargs.get("num_workers")
        batch_size = num_workers if num_workers is not None else NUM_CPU
    else:
        kwargs.pop("batch_size")
    for i in range(0, len(images) - 1, batch_size):
        try:
            results = model.predict(
                source=[
                    np.ascontiguousarray(image[0])
                    for image in images[i : i + batch_size]
                ],
                conf=conf,
                stream=True,
                verbose=False,
                **kwargs,
            )
            for j, result in enumerate(results):
                yield get_result_stats(result), images[i + j][1]
        except Exception as e:
            print(e, file=sys.stderr)
            yield None


def predict_on_image(model, image, conf=0.6, **kwargs):
    """
    Predicts on a single image using the specified model.

    Parameters:
        model: Model
            The YOLO model instance.
        image: np.ndarray
            Image to predict on.
        conf: float, optional
            Confidence threshold for predictions. Default is 0.6.
        kwargs: dict
            Additional arguments for the model's predict function.

    Returns:
        ResultStats: Processed prediction results.
    """
    result = model.predict(
        image,
        conf=conf,
        **kwargs,
    )[0]
    return get_result_stats(result)


def predict_geotiff(model, geotiff_path, confidence, tile_size, imgsz, **kwargs):
    """
    Predicts on GeoTIFF data using a YOLO model.

    Parameters:
        model: YOLOModel
            The YOLO model instance.
        geotiff_path: str
            Path to the GeoTIFF file.
        confidence: float
            Confidence threshold for predictions.
        tile_size: int
            Size of tiles for processing.
        imgsz: Tuple[int, int]
            Image size for YOLO model input.
        kwargs: dict
            Additional arguments for prediction.

    Returns:
        Tuple[List[Results], GeoDataFrame]:
            Detection results and processed GeoDataFrame with geometries.
    """
    tiles, epsg_code = tile_raster_and_convert_to_png(geotiff_path, tile_size=tile_size)
    results = []

    pbar = tqdm(total=len(tiles), desc="Detections 0", leave=False)
    for result in predict_on_image_stream(
        model, imgsz=imgsz, images=tiles, conf=confidence, **kwargs
    ):
        if result is not None and result[0][1][1] is not None:
            results.append(result)
            pbar.set_description(f"Detections {len(results)}")
        pbar.update()
    pbar.update()
    pbar.close()

    columns = [
        "subregion",
        "start_year",
        "end_year",
        "path",
        "class_id",
        "class_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "geometry",
    ]
    rows = []
    geometry = []

    subregion, years = parse_subregion_and_years_from_path(geotiff_path)

    with rasterio.open(geotiff_path) as src:
        for (result, data), coords in results:
            row, col = src.index(*coords)
            for mask in data[1]:  # Assuming this is the segmentation mask
                polygons = mask_to_polygon(mask)

                class_id = data[2][0]
                class_name = result.names[class_id]
                for bbox in data[0]:
                    bbox_x, bbox_y, bbox_xx, bbox_yy = (
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                    )
                    bbox_y, bbox_x = src.xy(row + bbox_y, col + bbox_x)
                    bbox_yy, bbox_xx = src.xy(row + bbox_yy, col + bbox_xx)
                    bbox_w = bbox_xx - bbox_x
                    bbox_h = bbox_yy - bbox_y
                    for unioned_geometry in polygons:
                        unioned_geometry = Polygon(
                            [
                                src.xy(row + y, col + x)
                                for x, y in unioned_geometry.exterior.coords
                            ]
                        )
                        rows.append(
                            {
                                "subregion": subregion,
                                "start_year": years[0],
                                "end_year": years[1],
                                "path": geotiff_path,
                                "class_id": int(class_id),
                                "class_name": class_name,
                                "bbox_x": bbox_x,
                                "bbox_y": bbox_y,
                                "bbox_w": bbox_w,
                                "bbox_h": bbox_h,
                            }
                        )
                        geometry.append(unioned_geometry.buffer(0))

    gdf = gpd.GeoDataFrame(
        rows, columns=columns, geometry=geometry, crs=f"EPSG:{epsg_code}"
    )

    if gdf.empty:
        return results, gdf

    # Union all intersecting polygons
    unioned_geometry = unary_union(gdf["geometry"])

    distinct_geometries = []
    if isinstance(unioned_geometry, MultiPolygon):
        for poly in unioned_geometry.geoms:
            poly = normalize(poly)
            distinct_geometries.append(poly)
    else:
        unioned_geometry = normalize(unioned_geometry)
        geometry.append(unioned_geometry)

    rows = []
    geometry = []
    for geom in distinct_geometries:
        # Find rows in the original gdf that match this unionized geometry
        matching_rows = gdf[
            gdf["geometry"].intersects(geom)
        ]  # Find all original rows that intersect this new geometry

        # Add a new row for the unioned geometry, keeping other relevant information from the first matching row
        if not matching_rows.empty:
            row_to_keep = matching_rows.iloc[
                0
            ].copy()  # Copy the first matching row to keep its other fields
            # Update the geometry to the unionized one
            rows.append(row_to_keep)
            geometry.append(geom)
    gdf_unionized = gpd.GeoDataFrame(
        rows, columns=columns, geometry=geometry, crs=gdf.crs
    )

    return results, gdf_unionized


def predict_geotiffs(
    model, geotiff_paths, *, confidence, tile_size, imgsz, max_images=2, **kwargs
):
    """
    Predicts geospatial data from a list of GeoTIFF file paths using a given model.

    Parameters:
        model: The model used for predictions (e.g., a machine learning model).
        geotiff_paths (list of str): A list of paths to the GeoTIFF files.
        confidence (float): The confidence threshold for predictions.
        tile_size (int): The size of tiles to split the GeoTIFF into for processing.
        imgsz (int): The image size to resize the GeoTIFF tiles to.
        max_images (int, optional): Maximum number of images to process in parallel. Defaults to 2.
        **kwargs: Additional keyword arguments passed to the `predict_geotiff` function.

    Returns:
        tuple: A tuple containing:
            - results (list): A list of the prediction results for each GeoTIFF.
            - gdfs (list): A list of GeoDataFrames containing geospatial data.

    Example:
        results, gdfs = predict_geotiffs(model, ["path/to/file1.tif", "path/to/file2.tif"], confidence=0.5, tile_size=256, imgsz=512)
    """
    results = []
    gdfs = []

    def get_index_with_crs(gdf):
        for i, g in enumerate(gdfs):
            if g.crs == gdf.crs:
                return i
        return -1

    pbar = trange(len(geotiff_paths), desc="Processing predictions", leave=False)

    with ThreadPoolExecutor(max_workers=max_images) as executor:
        futures = [
            executor.submit(
                predict_geotiff,
                model,
                path,
                confidence,
                tile_size=tile_size,
                imgsz=imgsz,
                **kwargs,
            )
            for path in geotiff_paths
        ]
        for future in as_completed(futures):
            if future.exception() is not None:
                print(future.exception(), file=sys.stderr)
            else:
                result, _gdf = future.result()
                results.append(result)
                if len(gdfs) == 0:
                    gdfs.append(_gdf)
                elif not _gdf.empty:
                    index = get_index_with_crs(_gdf)
                    if index == -1:
                        gdfs.append(_gdf)
                    else:
                        gdfs[index] = gpd.GeoDataFrame(
                            pd.concat([gdfs[index], _gdf], ignore_index=True),
                            crs=_gdf.crs,
                        )
            pbar.update()
    pbar.close()

    return results, gdfs
