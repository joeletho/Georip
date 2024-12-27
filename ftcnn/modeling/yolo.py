import os
import random
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import skimage.io as io
from matplotlib import pyplot as plt
from pandas.compat import sys
from shapely import MultiPolygon, Polygon, normalize, unary_union
from tqdm.auto import tqdm, trange

from ftcnn.datasets.tools import make_ndvi_difference_dataset
from ftcnn.geometry import stringify_points
from ftcnn.geospacial.utils import encode_classes, encode_default_classes
from ftcnn.io import (clear_directory, collect_files_with_suffix, save_as_csv,
                      save_as_gpkg, save_as_shp)
from ftcnn.raster.tools import tile_raster_and_convert_to_png
from ftcnn.utils import NUM_CPU

from .utils import (BBox, XYInt, YOLODataset,
                    extract_annotated_label_and_image_data, write_classes)


def plot_yolo_results(
    results, *, shape: XYInt | None = None, figsize: XYInt | None = None
):
    """
    Plots YOLO results in a grid layout.

    Parameters:
        results: List[YOLOResult]
            A list of YOLO result objects containing image and detection data to be plotted.
        shape: Tuple[int, int], optional
            The shape (rows, columns) of the plot grid. Defaults to (1, len(results)).
        figsize: Tuple[int, int], optional
            The size of the figure in inches (width, height). Defaults to None.

    Returns:
        None

    Raises:
        Exception: If the number of results exceeds the grid shape.
    """
    if shape is None:
        shape = (1, len(results))

    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    elif shape[0] == 1:
        axes = np.array(axes)
    axes = axes.ravel()

    if len(axes) < len(results):
        raise Exception(
            "Invalid shape: number of results exceeds the shape of the plot"
        )

    for i, r in enumerate(results):
        img = r.plot()
        axes[i].imshow(img)
    plt.show()


def yolo_make_dataset(labels_path, images_path, class_map, root_dir):
    """
    Prepares a YOLO dataset by copying relevant labels and images into a structured directory.

    Parameters:
        labels_path: PathLike
            Path to the directory containing label files.
        images_path: PathLike
            Path to the directory containing image files.
        class_map: Dict[int, str]
            Mapping of class IDs to class names.
        root_dir: PathLike
            Path to the root directory where the YOLO dataset will be created.

    Returns:
        None

    Raises:
        Exception: If copying labels and images fails.
    """
    root_dir = Path(root_dir).resolve()
    labels_dir = root_dir / "labels"
    images_dir = root_dir / "images"

    pbar = trange(3, desc="Preparing root directory", leave=False)

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    pbar.update()

    clear_directory(labels_dir)
    pbar.update()
    clear_directory(images_dir)
    pbar.update()
    pbar.close()

    for id in tqdm(class_map.keys(), desc="Copying to root directory", leave=False):
        yolo_copy_labels_and_images_containing_class(
            str(id),
            src_labels_dir=labels_path,
            src_images_dir=images_path,
            dest_dir=root_dir,
        )
    pbar = trange(1, desc="Preparing labels and classes")
    classes = [str(id) for id in list(class_map.keys())]
    print(classes)
    yolo_remove_annotations_not_in(classes, labels_dir=labels_dir)

    pbar.update()
    classes, _ = yolo_recategorize_classes(
        class_map,
        labels_dir,
    )
    write_classes(classes, root_dir / "classes.txt")
    pbar.set_description("Complete")
    pbar.close()


def yolo_get_labels_and_images(
    label_paths, image_paths, class_map, *, num_workers=None, from_format="yolo"
):
    """
    Retrieves labels and images associated with specified classes.

    Parameters:
        label_paths: PathLike
            Directory containing label files.
        image_paths: PathLike
            Directory containing image files.
        class_map: Dict[int, str]
            Mapping of class IDs to class names.
        num_workers: int, optional
            Number of worker threads to use. Defaults to the number of CPUs.
        from_format: str, optional
            Format of the label files ("yolo" by default).

    Returns:
        Tuple[List[Labels], List[Images]]
            Lists of labels and images.

    Raises:
        Exception: If label and image paths do not align or if any label fails to load.
    """
    labels = []
    images = []

    def __remove_not_in__(sources, targets):
        results = []
        for spath in tqdm(sources, desc="Cleaning unused sources", leave=False):
            for tpath in targets:
                if spath.stem == tpath.stem:
                    results.append(spath)
                    break
        return results

    def __preprocess_paths__(a, b):
        pbar = trange(2, desc="Preprocessing paths", leave=False)
        a = [Path(a, p) for p in os.listdir(a)]
        a.sort(key=lambda p: p.stem)
        b = [Path(b, p) for p in os.listdir(b)]
        b.sort(key=lambda p: p.stem)
        pbar.update()

        a = __remove_not_in__(a, b)
        b = __remove_not_in__(b, a)
        pbar.update()

        if len(a) != len(b):
            raise Exception(
                "Provided paths to not map. Each label path must have a associated image path"
            )
        pbar.close()
        return a, b

    def __collect_labels_and_images__(lpaths, ipaths, classes):
        lbls = []
        imgs = []
        if len(lpaths) != len(ipaths):
            raise Exception("Path lists must have the same length")

        for i in trange(len(lpaths), desc="Collecting labels and images", leave=False):
            if lpaths[i].stem != ipaths[i].stem:
                raise Exception(f"Path stems at index {i} do not match")

            extracted_labels, image = extract_annotated_label_and_image_data(
                lpaths[i], ipaths[i], classes
            )
            if from_format != "yolo":
                for label in extracted_labels:
                    bbox = label.bbox
                    bbox.x = (bbox.x + bbox.width) / 2
                    bbox.y = (bbox.y + bbox.height) / 2

            lbls.extend(extracted_labels)
            imgs.append(image)
            sleep(0.1)
        return lbls, imgs

    label_paths, image_paths = __preprocess_paths__(label_paths, image_paths)
    if num_workers is None:
        num_workers = NUM_CPU
    batch = len(label_paths) // num_workers

    pbar = trange(num_workers, desc="Progress")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(label_paths), batch):
            futures.append(
                executor.submit(
                    __collect_labels_and_images__,
                    label_paths[i : i + batch],
                    image_paths[i : i + batch],
                    class_map,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            labels.extend(result[0])
            images.extend(result[1])
            pbar.update()

    pbar.set_description("Complete")
    pbar.close()

    return labels, images


def yolo_copy_labels_and_images_containing_class(
    class_id, *, src_labels_dir, src_images_dir, dest_dir
):
    """
    Copies labels and images containing a specific class to a destination directory.

    Parameters:
        class_id: str
            The class ID to filter and copy.
        src_labels_dir: PathLike
            Source directory containing label files.
        src_images_dir: PathLike
            Source directory containing image files.
        dest_dir: PathLike
            Destination directory to store filtered labels and images.

    Returns:
        None

    Raises:
        IOError: If copying files fails.
    """
    label_paths = []
    image_paths = []
    labels_dest = Path(dest_dir, "labels").resolve()
    images_dest = Path(dest_dir, "images").resolve()
    class_id = str(class_id)

    for filename in tqdm(
        os.listdir(src_labels_dir), desc="Collecting label paths", leave=False
    ):
        label_path = Path(src_labels_dir, filename).resolve()
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) and parts[0] == class_id:
                    label_paths.append(label_path)
                    break
    labels = {p.stem: p for p in label_paths}
    for filename in tqdm(
        os.listdir(src_images_dir), desc="Collecting image paths", leave=False
    ):
        stem = os.path.splitext(filename)[0]
        if labels.get(stem):
            image_paths.append(Path(src_images_dir, filename))

    images = {p.stem: p for p in image_paths}
    for stem, p in labels.items():
        if not images.get(stem):
            label_paths.remove(p)

    for label in tqdm(label_paths, desc="Copying lables", leave=False):
        shutil.copy(label, labels_dest / label.name)

    for image in tqdm(image_paths, desc="Copying images", leave=False):
        shutil.copy(image, images_dest / image.name)
    print(f"Complete. Copied {len(label_paths)} labels and images")


def yolo_remove_annotations_not_in(class_ids, *, labels_dir):
    """
    Removes annotations in label files that do not match specified class IDs.

    Parameters:
        class_ids: List[str]
            A list of valid class IDs to retain.
        labels_dir: PathLike
            Directory containing label files.

    Returns:
        None
    """
    labels_dir = Path(labels_dir).resolve()
    files_annotations = {}
    filenames = os.listdir(labels_dir)
    for filename in tqdm(filenames, desc="Collecting class annotations"):
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        with open(path) as f:
            for line in f:
                if len(line) == 0:
                    continue
                parts = line.split()
                if parts[0] in class_ids:
                    if not files_annotations.get(filename):
                        files_annotations[filename] = []
                    files_annotations[filename].append(line)

    for filename in tqdm(filenames, desc="Writing to files"):
        annotations = files_annotations.get(filename)
        if not annotations:
            continue
        lines = "\n".join(line for line in annotations)
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        with open(path, "w") as f:
            f.write(lines)

    print(f"Complete. {len(files_annotations.keys())} files written")


def yolo_recategorize_classes(classes: dict, labels_dir):
    """
    Recategorizes class IDs in label files and maps old IDs to new IDs.

    Parameters:
        classes: Dict[str, str]
            Dictionary mapping old class IDs to new class names.
        labels_dir: PathLike
            Directory containing label files.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]
            Updated class dictionary and mapping of old to new IDs.

    Raises:
        Exception: If recategorization fails.
    """
    labels_dir = Path(labels_dir).resolve()
    old_new_map = {}
    for filename in tqdm(os.listdir(labels_dir), desc="Collecting class ids"):
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        with open(path) as f:
            for line in f:
                line.strip()
                parts = line.split()
                if len(parts) == 0:
                    continue
                id = str(parts[0])
                if id not in old_new_map.keys():
                    old_new_map[id] = None
    for i, id in enumerate(old_new_map.keys()):
        old_new_map[str(id)] = str(i)

    for filename in tqdm(os.listdir(labels_dir), desc="Writing to files"):
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        lines = []
        with open(path, "r+") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 0:
                    continue
                id = str(parts[0])
                if id in [str(k) for k in old_new_map.keys()]:
                    parts[0] = old_new_map[id]
                line = " ".join(part for part in parts)
                if len(line) > 0:
                    lines.append(line)
            f.truncate(0)
            f.seek(0)
            f.write("{}\n".format("\n".join(line for line in lines)))

    print(old_new_map)
    new_classes = {}
    for old, new in old_new_map.items():
        name = classes.get(str(old))
        new_classes[str(new)] = name

    for name in new_classes.values():
        if name == "None":
            raise Exception(f"Class assignment failed: {new_classes}")

    print("Complete")

    return new_classes, old_new_map


def convert_xml_bbox_to_yolo(df: pd.DataFrame):
    """
    Converts bounding boxes from XML format to YOLO format in a DataFrame.

    Parameters:
        df: pd.DataFrame
            DataFrame containing XML bounding box information.

    Returns:
        None
    """
    pbar = tqdm(
        total=df.shape[0], desc="Converting XML BBox to YOLO format", leave=False
    )
    for _, row in df.iterrows():
        bbox = BBox(
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        )

        bbox.width -= bbox.x
        bbox.height -= bbox.y

        row["bbox_x"] = bbox.x
        row["bbox_y"] = bbox.y
        row["bbox_w"] = bbox.width
        row["bbox_h"] = bbox.height
        pbar.update()
    pbar.close()


def convert_xml_dataframe_to_yolo(df: pd.DataFrame):
    """
    Converts a DataFrame from XML format to YOLO format.

    Parameters:
        df: pd.DataFrame
            DataFrame with XML-style columns.

    Returns:
        None
    """
    df.rename(
        columns={
            "filename": "filename",
            "name": "class_name",
            "width": "width",
            "height": "height",
            "xmin": "bbox_x",
            "ymin": "bbox_y",
            "xmax": "bbox_w",
            "ymax": "bbox_h",
        },
        inplace=True,
    )


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


def get_result_stats(result):
    """
    Extracts detection and segmentation statistics from YOLO results.

    Parameters:
        result: YOLOResult
            The result object from YOLO model prediction.

    Returns:
        Tuple[YOLOResult, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            Processed detection and segmentation results.
    """
    # Detection
    classes = result.boxes.cls.cpu().numpy()  # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()  # box with xyxy format, (N, 4)

    # Segmentation
    if result.masks is None:
        masks = None
    else:
        masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)

    return result, (boxes, masks, classes, probs)


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


def parse_subregion_and_years_from_path(image_path):
    """
    Parses the subregion and years from the file path.

    Parameters:
        image_path: str
            Path to the image file.

    Returns:
        Tuple[str, Tuple[int, int]]:
            Subregion and start-end year range.
    """
    parts = Path(image_path).stem.split("_")
    subregion = parts[0]
    years = parts[1]
    if "extended" in years.lower():
        subregion = subregion + "E"
        years = parts[2]
    elif subregion[-2:].isnumeric():
        start = 0
        while start < len(subregion) and not subregion[start].isdigit():
            start += 1
        if start >= len(subregion):
            raise ValueError(f"Error parsing years from {image_path}")
        years = subregion[start:]
        subregion = subregion[:start]
    years = years.split("to")
    return subregion, (int(years[0]), int(years[1]))


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
    tiles, epsg_code = tile_raster_and_convert_to_png(
        geotiff_path, tile_size=tile_size
    )
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


def ndvi_to_yolo_dataset(
    shp_file,
    ndvi_dir,
    output_dir,
    *,
    years=None,
    start_year_col="start_year",
    end_year_col="end_year",
    geom_col="geometry",
    tile_size=None,
    clean_dest=False,
    xy_to_index=True,
    encoder=encode_default_classes,
    exist_ok=False,
    save_csv=False,
    save_shp=False,
    save_gpkg=False,
    ignore_empty_geom=True,
    generate_labels=True,
    tif_to_png=True,
    use_segments=True,
    generate_train_data=True,
    split=0.75,
    split_mode="all",
    shuffle_split=True,
    shuffle_background=True,
    background_bias=None,
    pbar_leave=True,
    num_workers=None,
):
    """
    Converts NDVI (Normalized Difference Vegetation Index) data into a YOLO-compatible dataset format.

    Parameters:
        shp_file (str): Path to the shapefile containing the polygons.
        ndvi_dir (str): Directory containing the NDVI image files.
        output_dir (str): Directory where the output dataset will be saved.
        years (list of int, optional): A list of years to process. Defaults to None.
        start_year_col (str, optional): Column name for the start year. Defaults to "start_year".
        end_year_col (str, optional): Column name for the end year. Defaults to "end_year".
        geom_col (str, optional): Column name for the geometry data. Defaults to "geometry".
        tile_size (int, optional): Size of the tiles for the NDVI data. Defaults to None.
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.
        xy_to_index (bool, optional): Whether to convert coordinates to an index. Defaults to True.
        encoder (function, optional): Function for encoding class labels. Defaults to `encode_default_classes`.
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to False.
        save_csv (bool, optional): Whether to save the dataset as a CSV file. Defaults to False.
        save_shp (bool, optional): Whether to save the dataset as a shapefile. Defaults to False.
        save_gpkg (bool, optional): Whether to save the dataset as a geopackage. Defaults to False.
        ignore_empty_geom (bool, optional): Whether to ignore empty geometries. Defaults to True.
        generate_labels (bool, optional): Whether to generate labels for the dataset. Defaults to True.
        tif_to_png (bool, optional): Whether to convert TIFF images to PNG format. Defaults to True.
        use_segments (bool, optional): Whether to use segments in the dataset. Defaults to True.
        generate_train_data (bool, optional): Whether to generate training data. Defaults to True.
        split (float, optional): Proportion of data to use for training. Defaults to 0.75.
        split_mode (str, optional): Mode of data splitting ("all", "random", etc.). Defaults to "all".
        shuffle_split (bool, optional): Whether to shuffle the data when splitting. Defaults to True.
        shuffle_background (bool, optional): Whether to shuffle background images. Defaults to True.
        background_bias (float, optional): Bias factor for background data. Defaults to None.
        min_labels_required (int, optional): Minimum number of labels required for a valid dataset. Defaults to 10.
        pbar_leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
        num_workers (int, optional): Number of worker processes for parallel processing. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - yolo_ds (YOLODataset): The YOLO dataset object.
            - train_data (optional): The training data if generated.

    Example:
        yolo_ds, train_data = ndvi_to_yolo_dataset("path/to/shapefile.shp", "path/to/ndvi_dir", "path/to/output_dir", years=[2020, 2021], generate_labels=True)
    """
    ignore_empty_geom = ignore_empty_geom and background_bias is None

    gdf, (meta_dir, tiles_dir, output_fname) = make_ndvi_difference_dataset(
        shp_file,
        ndvi_dir,
        output_dir,
        years=years,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        geom_col=geom_col,
        tile_size=tile_size,
        clean_dest=clean_dest,
        xy_to_index=xy_to_index,
        exist_ok=exist_ok,
        save_csv=save_csv,
        save_shp=save_shp,
        save_gpkg=False,
        ignore_empty_geom=ignore_empty_geom,
        tif_to_png=tif_to_png,
        pbar_leave=False,
        num_workers=num_workers,
    )

    csv_dir = meta_dir / "csv"
    shp_dir = meta_dir / "shp"

    n_calls = 3
    n_calls += 1 if generate_labels else 0
    n_calls += 1 if generate_train_data else 0
    pbar = trange(
        n_calls, desc="Creating YOLO dataset - Encoding classes", leave=pbar_leave
    )

    gdf = encode_classes(gdf, encoder)

    labeled_images = gdf.loc[gdf["class_id"] != -1].values.tolist()

    if ignore_empty_geom or background_bias is None:
        new_rows = labeled_images
    else:
        background_images = gdf.loc[gdf["class_id"] == -1].values.tolist()
        if shuffle_background:
            random.shuffle(background_images)
        background_images = background_images[
            : int(len(labeled_images) * background_bias)
        ]

        new_rows = labeled_images + background_images

    gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)

    if save_csv or save_shp:
        output_fname = Path(f"{output_fname}_encoded")
        if save_csv:
            save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / output_fname.with_suffix(".shp"),
            )
        if save_gpkg:
            save_as_gpkg(
                gdf,
                shp_dir / output_fname.with_suffix(".gpkg"),
            )
    pbar.update()

    pbar.set_description(
        f"Creating YOLO dataset - Creating YOLODataset with {len(gdf)} labels"
    )
    yolo_ds = to_yolo(gdf)
    pbar.set_description("Creating YOLO dataset - Dataset created")
    pbar.update()

    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    yolo_ds.generate_yaml_file(
        root_abs_path=output_dir,
        dest_abs_path=output_dir / "config",
        train_path="images/train",
        val_path="images/val",
    )

    train_data = None
    if generate_labels or generate_train_data:
        pbar.update()
        pbar.set_description("Creating YOLO dataset - Generating labels")

        yolo_ds.generate_label_files(
            dest_path=output_dir / "labels" / "generated",
            clear_dir=clean_dest,
            overwrite_existing=exist_ok,
            use_segments=use_segments,
        )
        if generate_train_data:
            pbar.update()
            pbar.set_description(
                "Creating YOLO dataset - Splitting dataset and copying files"
            )

            ds_images_dir = (
                output_dir / "images" / "png-tiles" if tif_to_png else tiles_dir
            )
            train_data = yolo_ds.split_data(
                images_dir=ds_images_dir,
                labels_dir=output_dir / "labels" / "generated",
                split=split,
                shuffle=shuffle_split,
                recurse=True,
                mode=split_mode,
            )

            yolo_df = yolo_ds.data_frame
            yolo_ds.compile(NUM_CPU)
            yolo_ds.data_frame = yolo_df

    if save_csv:
        yolo_ds.to_csv(csv_dir / "yolo_ds.csv")

    pbar.update()
    pbar.set_description("Complete")
    pbar.close()

    return yolo_ds, train_data


def to_yolo(gdf: gpd.GeoDataFrame, compile=True) -> YOLODataset:
    """
    Converts a GeoDataFrame into a YOLO dataset format.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing the labeled data.
        compile (bool, optional): Whether to compile the dataset. Defaults to True.

    Returns:
        YOLODataset: The resulting YOLO dataset.

    Example:
        yolo_ds = to_yolo(gdf)
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: stringify_points(x.exterior.coords)
    )
    tmp_path = "/tmp/ftcnn_yolo_ds.csv"
    gdf.to_csv(tmp_path)
    try:
        ds = YOLODataset.from_csv(
            tmp_path,
            segments_key="geometry",
            convert_bounds_to_bbox=True,
            num_workers=NUM_CPU,
            compile=compile,
        )
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    except Exception as e:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise e
    return ds


def draw_yolo_bboxes(image_path, label_path, class_names=None):
    """
    Draws YOLO-style bounding boxes on an image based on the provided label file.

    Parameters:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO label file (text file containing bounding boxes).
        class_names (list of str, optional): A list of class names for labeling the bounding boxes. Defaults to None.

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.

    Example:
        img_with_bboxes = draw_yolo_bboxes("image.jpg", "image.txt", class_names=["class1", "class2"])
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Read the YOLO-formatted label file
    with open(label_path, "r") as f:
        bboxes = f.readlines()

    # Loop through each line (bounding box) in the label file
    for bbox in bboxes:
        bbox = bbox.strip().split()

        class_id = int(bbox[0])  # Class ID is the first value
        x_center = float(bbox[1])  # YOLO X center (relative to image width)
        y_center = float(bbox[2])  # YOLO Y center (relative to image height)
        bbox_width = float(bbox[3])  # YOLO width (relative to image width)
        bbox_height = float(bbox[4])  # YOLO height (relative to image height)

        # Convert YOLO coordinates back to absolute pixel values
        x_center_abs = int(x_center * img_width)
        y_center_abs = int(y_center * img_height)
        bbox_width_abs = int(bbox_width * img_width)
        bbox_height_abs = int(bbox_height * img_height)

        # Calculate the top-left corner of the bounding box
        x_min = int(x_center_abs - (bbox_width_abs / 2))
        y_min = int(y_center_abs - (bbox_height_abs / 2))
        x_max = int(x_center_abs + (bbox_width_abs / 2))
        y_max = int(y_center_abs + (bbox_height_abs / 2))

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Bounding box color (green)
        thickness = 2  # Thickness of the box
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Optionally, label the bounding box with the class name
        if class_names:
            label = class_names[class_id]
            cv2.putText(
                img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # Convert the image back to RGB for display with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def yolo_create_truth_and_prediction_pairs(
    truth_images_dir, truth_labels_dir, pred_images_dir
):
    """
    Creates pairs of ground truth and predicted images with YOLO bounding boxes for evaluation.

    Parameters:
        truth_images_dir (str): Directory containing the ground truth images.
        truth_labels_dir (str): Directory containing the ground truth YOLO label files.
        pred_images_dir (str): Directory containing the predicted images.

    Returns:
        list of tuple: A list of pairs of images (ground truth and predicted images).

    Example:
        images = yolo_create_truth_and_prediction_pairs("truth_images", "truth_labels", "pred_images")
    """
    truth_image_paths = collect_files_with_suffix([".jpg", ".png"], truth_images_dir)
    truth_label_paths = collect_files_with_suffix(".txt", truth_labels_dir)
    pred_image_paths = collect_files_with_suffix([".jpg", ".png"], pred_images_dir)

    assert (
        len(truth_image_paths) == len(truth_label_paths)
        and "Number of Images and labels must match"
    )
    assert (
        len(truth_image_paths) == len(pred_image_paths)
        and "Number of truth images must match predicted images"
    )

    # Align image and label paths
    for i in range(len(truth_image_paths)):
        found = False
        for j, label in enumerate(truth_label_paths):
            if truth_image_paths[i].stem == label.stem:
                truth_label_paths[i], truth_label_paths[j] = (
                    truth_label_paths[j],
                    truth_label_paths[i],
                )
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Could not find label for {truth_image_paths[i].name}"
            )

    # Create truth-bounded images and add it and its predicted counterpart to the list of images
    images = []
    for i in range(len(truth_image_paths)):
        found = False
        for pred_path in pred_image_paths:
            if truth_image_paths[i].stem == pred_path.stem:
                truth_image = draw_yolo_bboxes(
                    truth_image_paths[i], truth_label_paths[i]
                )
                pred_image = io.imread(pred_path)
                images.append((truth_image, pred_image))
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Could not find {truth_image_paths[i].name} in predicted images"
            )

    return images
