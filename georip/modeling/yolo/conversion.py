from time import time

import geopandas as gpd
import pandas as pd
from georip.datasets.utils import TMP_FILE_PREFIX
from georip.geometry import stringify_points
from georip.modeling.utils import (AnnotatedLabel, BBox,
                                  convert_segmented_to_bbox_annotation,
                                  parse_images_from_labels,
                                  parse_labels_from_dataframe)
from georip.modeling.yolo import YOLODatasetBase
from georip.utils import GEORIP_TMP_DIR, NUM_CPU
from PIL import Image
from tqdm.auto import tqdm, trange


def geodataframe_to_yolo(
    gdf: gpd.GeoDataFrame, geometry_column: str = "geometry", compile=True
) -> YOLODatasetBase:
    """
    Converts a GeoDataFrame into a YOLO dataset format.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame containing the labeled data.
        geometry_column (str): The name of the column containing geometry. Default is "geometry".
        compile (bool, optional): Whether to compile the dataset. Defaults to True.

    Returns:
        YOLODataset: The resulting YOLO dataset.

    Example:
        yolo_ds = geodataframe_to_yolo(gdf)
    """
    gdf = gdf.copy()
    gdf[geometry_column] = gdf[geometry_column].apply(
        lambda x: stringify_points(x.exterior.coords)
    )
    tmp_path = GEORIP_TMP_DIR / f"{TMP_FILE_PREFIX}yolo_ds_{time()}.csv"
    gdf.to_csv(tmp_path)
    num_workers = NUM_CPU

    try:
        ds = YOLODatasetBase.from_csv(
            tmp_path,
            segments_key=geometry_column,
            convert_bounds_to_bbox=True,
            num_workers=num_workers,
            compile=compile,
        )
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e
    return ds


def yolo_create_dataset_from_dataframe(df, *, imgdir, parallel=True):
    pbar = trange(3, position=0, desc="Building YOLO Dataset")

    labels = parse_labels_from_dataframe(df)
    pbar.update()

    images = parse_images_from_labels(labels, imgdir, parallel=parallel)
    pbar.update()

    ds = YOLODatasetBase(labels=labels, images=images)
    pbar.update()

    pbar.set_description("Complete")
    pbar.close()
    return ds


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


def convert_coco_label_to_yolo(label_path, image_path):
    labels = []
    errors = []

    converted_label = convert_segmented_to_bbox_annotation(label_path)

    for annotation in converted_label:
        label_parts = annotation.split()
        class_id = int(label_parts[0]) + 1

        with Image.open(image_path) as img:
            width, height = img.size

        bbox = label_parts[1:]
        bbox = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        bbox.x = width * (2 * bbox.width - bbox.x)
        bbox.y = height * (2 * bbox.height - bbox.y)
        bbox.width = width * bbox.width
        bbox.height = height * bbox.width

        labels.append(
            AnnotatedLabel(
                class_id=class_id, class_name="", bbox=bbox, image_filename=""
            )
        )
    return labels, errors
