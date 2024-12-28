import os

import geopandas as gpd
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm, trange

from ftcnn.geometry import stringify_points
from ftcnn.modeling.utils import (
    AnnotatedLabel,
    BBox,
    convert_segmented_to_bbox_annotation,
    parse_images_from_labels,
    parse_labels_from_dataframe,
)
from ftcnn.modeling.yolo import YOLODatasetBase
from ftcnn.utils import NUM_CPU


def geodataframe_to_yolo(gdf: gpd.GeoDataFrame, compile=True) -> YOLODatasetBase:
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
        ds = YOLODatasetBase.from_csv(
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
