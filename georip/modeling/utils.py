import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import ArgumentError
from enum import Enum
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import cv2
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import rasterio
import skimage
import supervision as sv
import torch
from matplotlib import pyplot as plt
from numpy import NaN
from PIL import Image
from torchvision.transforms import v2 as T
from tqdm.auto import tqdm, trange

from georip.io import collect_files_with_suffix, pathify
from georip.utils import StrPathLike

XYPair = tuple[float | int, float | int]
XYInt = tuple[int, int]
ClassMap = dict[str, str]


class DatasetSplitMode(Enum):
    All = "all"
    Collection = "collection"


class Serializable:
    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)


class BBox(Serializable):
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x: float = x
        self.y: float = y
        self.width: float = w
        self.height: float = h

    def __eq__(self, other):
        return isinstance(other, BBox) and hash(self) == hash(other)

    def __hash__(self):
        return hash((self.x, self.y, self.width, self.height))


class ImageData(Serializable):
    def __init__(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix == ".tif":
            image = rasterio.open(filepath)
            self.shape: XYPair = image.shape
            self.bounds = image.bounds
            self.transform = image.transform
            image.close()
            self.filename: str = filepath.name
        else:
            with Image.open(filepath) as image:
                self.shape: XYPair = image.size
                self.filename: str = filepath.name

        self.basename: str = filepath.stem
        self.extension: str = filepath.suffix
        self.filepath: str = str(filepath)

    def __eq__(self, other):
        return isinstance(other, ImageData) and hash(self) == hash(other)

    def __hash__(self):
        return hash(
            (
                self.shape[0],
                self.shape[1],
                self.filepath,
            )
        )


class AnnotatedLabel(Serializable):
    def __init__(
        self,
        *,
        type: str | None = "",
        class_id: int | None = None,
        class_name: str,
        bbox: BBox | None = None,
        segments: list[float] | None = None,
        image_filename: str,
        filepath: str | Path | None = None,
    ):
        self.type = type
        self.class_id: int | None = class_id
        self.class_name: str = class_name
        self.bbox: BBox | None = bbox
        self.image_filename: str = image_filename
        self.segments: list[float] | None = segments
        self.filepath: str | Path | None = str(filepath)

    def __eq__(self, other):
        return isinstance(other, AnnotatedLabel) and hash(self) == hash(other)

    def __hash__(self):
        return hash(
            (
                self.type,
                self.class_id,
                self.class_name,
                self.bbox,
                str(self.segments),
                self.image_filename,
                self.filepath,
            )
        )

    @staticmethod
    def parse_label(line_from_file: str):
        parts = line_from_file.split()
        class_id = parts[0]
        points = [float(point) for point in parts[1:]]
        if len(points) > 4:
            bbox = convert_segment_to_bbox(points)
            seg = points
        else:
            bbox = BBox(points[0], points[1], points[2], points[3])
            seg = []
        return AnnotatedLabel(
            class_id=int(class_id),
            class_name="",
            image_filename="",
            bbox=bbox,
            segments=seg,
        )

    @staticmethod
    def from_file(filepath: str | Path, image_filename="", class_name=""):
        filepath = Path(filepath)
        annotations = []
        with open(filepath) as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(0)
                for line in f:
                    label = AnnotatedLabel.parse_label(line)
                    label.filepath = str(filepath)
                    label.image_filename = image_filename
                    label.class_name = class_name
                    annotations.append(label)
            else:
                annotations.append(
                    AnnotatedLabel(
                        filepath=filepath,
                        class_name=class_name,
                        image_filename=image_filename,
                    )
                )

        return annotations


class XMLTree:
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.tree = ET.parse(filepath)

    def root(self):
        return self.tree.getroot()


def copy_split_data(
    ds: sv.DetectionDataset, label_map: ClassMap, images_dest_path, labels_dest_path
):
    images_dest_path = str(Path(images_dest_path).resolve())
    labels_dest_path = str(Path(labels_dest_path).resolve())

    os.makedirs(images_dest_path, exist_ok=True)
    os.makedirs(labels_dest_path, exist_ok=True)

    nfiles = len(ds.image_paths)
    pbar = tqdm(total=nfiles, desc="Copying labels and images")
    for path in ds.image_paths:
        path = Path(path)
        image_name = path.name
        image_stem = path.stem
        if label_map.get(image_stem):
            shutil.copyfile(
                label_map[image_stem],
                Path(labels_dest_path, f"{image_stem}.txt"),
            )
            shutil.copyfile(path, Path(images_dest_path, image_name))
        else:
            print(f"Key Error: key '{image_stem}' not found in labels", file=sys.stderr)

        pbar.update()
    pbar.set_description("Complete")
    pbar.close()


def copy_images_and_labels(image_paths, label_paths, images_dest, labels_dest):
    images_dest = Path(images_dest).resolve()
    labels_dest = Path(labels_dest).resolve()

    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(path) for path in image_paths]
    label_paths = [Path(path) for path in label_paths]

    nfiles = len(image_paths)
    pbar = tqdm(total=nfiles, desc="Copying labels and images", leave=False)

    for image_path in image_paths:
        found_label = False
        for label_path in label_paths:
            if label_path.stem == image_path.stem:
                shutil.copyfile(
                    label_path,
                    labels_dest / label_path.name,
                )
                shutil.copyfile(image_path, images_dest / image_path.name)
                found_label = True
                break
        if not found_label:
            print(
                f"Key Error: key '{image_path.stem}' not found in labels",
                file=sys.stderr,
            )
        pbar.update()
    pbar.close()


def remove_rows(df, pred_key: Callable):
    columns = df.columns
    rows_removed = []
    rows_kept = []
    pbar = trange(len(df) + 1, desc="Cleaning dataframe", leave=False)
    for _, row in df.iterrows():
        if pred_key(row):
            rows_kept.append(row)
        else:
            rows_removed.append(row)
        pbar.update()
    df_cleaned = pd.DataFrame(rows_kept, columns=columns, index=range(len(rows_kept)))
    df_removed = pd.DataFrame(
        rows_removed, columns=columns, index=range(len(rows_removed))
    )
    pbar.update()
    pbar.close()
    return df_cleaned, df_removed


def parse_labels_from_csv(
    csvpath,
    *,
    type_key="type",
    class_id_key="class_id",
    class_name_key="class_name",
    image_filename_key="filename",
    bbox_key="bbox",
    bbox_x_key="bbox_x",
    bbox_y_key="bbox_y",
    bbox_width_key="bbox_w",
    bbox_height_key="bbox_h",
    segments_key="segments",
    convert_bounds_to_bbox=False,
    ignore_errors=False,
):
    key_map = {
        "type": type_key,
        "class_id": class_id_key,
        "class_name": class_name_key,
        "filename": image_filename_key,
        "bbox": bbox_key,
        "bbox_x": bbox_x_key,
        "bbox_y": bbox_y_key,
        "bbox_w": bbox_width_key,
        "bbox_h": bbox_height_key,
        "segments": segments_key,
    }

    labels = []
    df = pd.read_csv(csvpath)

    for i, row in df.iterrows():
        data = {key: None for key in key_map.keys()}
        for key, user_key in key_map.items():
            try:
                field = str(row[user_key]).strip()
                data[key] = (
                    None
                    if len(field) == 0 or field.lower() == "nan" or field == "None"
                    else field
                )
            except Exception:
                pass

        if not ignore_errors:
            if data.get("class_id") is None:
                raise ValueError(f"Row index {i}: key '{class_id_key}' not found:", row)
            if data.get("class_name") is None:
                raise ValueError(
                    f"Row index {i}: key '{class_name_key}' not found:", row
                )
            if data.get("filename") is None:
                raise ValueError(
                    f"Row index {i}: key '{image_filename_key}' not found:", row
                )

        type = str(data["type"])
        if len(type) > 0 or not type[0].isalnum():
            type = "None"
        class_id = int(data["class_id"])
        class_name = str(data["class_name"])
        image_filename = str(data["filename"])
        bbox = None
        segments = None

        if data.get("bbox_x") is not None:
            bbox_x = float(data["bbox_x"])
            bbox_y = float(data["bbox_y"])
            bbox_width = float(data["bbox_w"])
            bbox_height = float(data["bbox_h"])
        elif data.get("bbox") is not None:
            _bbox = data["bbox"]
            if _bbox is NaN:
                data["bbox"] = None
            else:
                _bbox = _bbox.split()
                bbox_x = float(_bbox[0])
                bbox_y = float(_bbox[1])
                bbox_width = float(_bbox[2])
                bbox_height = float(_bbox[3])
        if data.get("bbox") or data.get("bbox_x"):
            if convert_bounds_to_bbox:
                bbox_width = bbox_width - bbox_x
                bbox_height = bbox_height - bbox_y
            bbox = BBox(bbox_x, bbox_y, bbox_width, bbox_height)

        if data.get("segments") is not None:
            segments = [float(val) for val in str(data["segments"]).split()]

        labels.append(
            AnnotatedLabel(
                type=type,
                class_id=class_id,
                class_name=class_name,
                image_filename=image_filename,
                bbox=bbox,
                segments=segments,
            )
        )

    return labels


def parse_images_from_labels(
    labels: list[AnnotatedLabel], imgdir: str | list[str], *, parallel: bool = True
):
    if isinstance(imgdir, "str"):
        imgdir = [imgdir]
    imgdir = [Path(dir).resolve() for dir in imgdir]
    image_files = [label.image_filename for label in labels]

    def __collect_images__(files, n=None):
        images = []
        for filename in tqdm(
            files,
            desc="Parsing images from labels" if n is None else f"{f'Batch {n}' : <12}",
            leave=False,
        ):
            found = False
            err = None
            for dir in imgdir:
                path = str(Path(dir, filename))
                try:
                    images.append(ImageData(path))
                    found = True
                except Exception as e:
                    err = e
                    pass
                if found:
                    break
            if not found:
                raise Exception("Could not find image", err)

        return images

    if not parallel:
        return __collect_images__(image_files, None)
    nfiles = len(labels)
    batch = 8
    chunksize = nfiles // batch
    images = []

    pbar = tqdm(total=batch, desc="Parsing images from labels", leave=False)
    with ThreadPoolExecutor() as exec:
        counter = 1
        futures = []
        i = 0
        while i < nfiles:
            start = i
            end = start + chunksize
            futures.append(
                exec.submit(__collect_images__, image_files[start:end], counter)
            )
            i = end
            counter += 1

        for future in as_completed(futures):
            for image in future.result():
                images.append(image)
            pbar.update()
    pbar.close()

    return images


def parse_labels_from_dataframe(df):
    labels = []
    with tqdm(
        total=df.shape[0], desc="Parsing labels from DataFrame", leave=False
    ) as pbar:
        for _, row in df.iterrows():
            try:
                class_id = int(row["class_id"])
            except Exception:
                class_id = -1

            labels.append(
                AnnotatedLabel(
                    class_id=class_id,
                    class_name=str(row["class_name"]),
                    image_filename=str(row["filename"]),
                    bbox=BBox(
                        float(row["bbox_x"]),
                        float(row["bbox_y"]),
                        float(row["bbox_w"]),
                        float(row["bbox_h"]),
                    ),
                )
            )
            pbar.update()
    pbar.close()

    return labels


def get_image_paths_from_labels(labels: list[AnnotatedLabel], imgdir):
    paths = []
    imgdir = Path(imgdir).resolve()
    for label in labels:
        path = None
        filename = label.image_filename

        if filename is not None:
            origin = Path(imgdir, filename)
            orig_stem = origin.stem
            for file in os.listdir(imgdir):
                filepath = Path(imgdir, file)
                if filepath.is_file() and filepath.stem == orig_stem:
                    path = filepath
                    break

        if path is None:
            raise Exception(f"Image path is missing '{filename}'")

        paths.append(path)

    return paths


def extract_annotated_label_and_image_data(label_path, path, class_map):
    labels = []

    image = ImageData(path)

    annotations = AnnotatedLabel.from_file(label_path)
    for label in annotations:
        label.class_name = class_map[str(label.class_id)]
        label.image_filename = image.filename
    labels.extend(annotations)
    return labels, image


def write_classes(class_map, dest):
    with open(dest, "w") as f:
        for id, name in sorted(class_map.items()):
            f.write(f"{id} {name}\n")


def extrapolate_annotations_from_label(label_path, path, class_map):
    labels = []
    errors = []

    converted_label = convert_segmented_to_bbox_annotation(label_path)
    classes = list(class_map.keys())

    for annotation in converted_label:
        label_parts = annotation.split()
        class_id = int(label_parts[0]) + 1

        if class_id not in classes:
            errors.append(f"Class id {class_id} in {label_path} not found. Skipping.")
            continue

        class_name = class_map[class_id].get("name")

        image_data = ImageData(path)

        width, height = image_data.shape

        bbox = label_parts[1:]
        bbox = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        bbox.x = width * (2 * bbox.width - bbox.x)
        bbox.y = height * (2 * bbox.height - bbox.y)
        bbox.width = width * bbox.width
        bbox.height = height * bbox.width

        labels.append(
            AnnotatedLabel(
                class_id=class_id,
                class_name=class_name,
                bbox=bbox,
                image_filename=image_data.filename,
                filepath=label_path,
            )
        )
    return labels, errors


def convert_segmented_to_bbox_annotation(file):
    labels = []
    lines = []
    with open(file) as f:
        for line in f:
            lines.append(line)
    for line in lines:
        parts = line.split()
        points = [float(point) for point in parts[1:]]
        points = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        bbox = convert_segment_to_bbox(points)
        labels.append(f"{parts[0]} {' '.join([str(p) for p in bbox])}")
    return labels


def convert_segment_to_bbox(points: list[float]):
    # If two adjacent coordinates are the same, we probably
    # have a set of edges.
    # Remove duplicate coords x, y, y, x, x, y -> x, y, x, y
    # hits = 0
    # for i in range(len(points) - 1):
    #     if points[i] == points[i + 1]:
    #         if hits > 1:
    #             points = list(set(points))
    #             break
    #         hits += 1

    n_points = len(points)
    xs = [points[i] for i in range(0, n_points, 2)]
    ys = [points[i] for i in range(1, n_points, 2)]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin
    bounds = [xmin, ymin, width, height]

    for b in bounds:
        if b < 0:
            raise ValueError("Point cannot be negative", bounds)

    return BBox(xmin, ymin, width, height)


def plot_hist(img, bins=64):
    hist, bins = skimage.exposure.histogram(img, bins)
    f, a = plt.subplots()
    a.plot(bins, hist)
    plt.show()


def make_dataset(
    images_dir: StrPathLike,
    labels_dir: StrPathLike,
    *,
    image_format: str | list[str] = ["jpg", "png"],
    label_format: str = "txt",
    train_split: float = 0.75,
    test_split: float = 0.1,
    stratify_split: bool = True,
    class_data: (
        dict[str, str | int] | None
    ) = None,  # Class data required if stratify_split is True
    mode: DatasetSplitMode | str = DatasetSplitMode.All,
    shuffle: bool = True,
    shuffle_seed: int | None = None,
    recurse: bool = True,
) -> tuple[
    tuple[list[ImageData], list[AnnotatedLabel]],  # Train
    tuple[list[ImageData], list[AnnotatedLabel]],  # Validation
    tuple[list[ImageData], list[AnnotatedLabel]],  # Test
]:
    """
    Creates and splits a dataset into train, validation, and test sets.

    Args:
        images_dir (StrPathLike): Directory containing image files.
        labels_dir (StrPathLike): Directory containing label files.
        image_format (str | list[str], optional): Image file format(s). Default is ["jpg", "png"].
        label_format (str, optional): Label file format. Default is "txt".
        train_split (float, optional): Proportion of data to use for training. Default is 0.75.
        test_split (float, optional): Proportion of data to use for testing. Default is 0.1.
        stratify_split (bool, optional): Whether to stratify splits by class. Default is True.
        class_data (dict[str, str | int] | None, optional): A dictionary mapping filenames (without extensions)
            to class identifiers for stratification. This is required if stratify_split is True.
            Example: {"image1": 0, "image2": 1}.
        mode (DatasetSplitMode | str, optional): Mode for splitting. Options are "all" or "collection".
            Default is DatasetSplitMode.All.
        shuffle (bool, optional): Whether to shuffle data before splitting. Default is True.
        shuffle_seed (int | None, optional): Seed for shuffling. Default is None.
        recurse (bool, optional): Whether to search directories recursively. Default is True.
        **kwargs: Additional keyword arguments to configure file formats.

    Returns:
        tuple: A tuple containing train, validation, and test data in the form of
        ((list[ImageData], list[AnnotatedLabel]), ...).

    Raises:
        FileNotFoundError: If no image or label files are found.
        ValueError: If stratify_split is True but class_data is not provided.
    """
    label_paths = collect_files_with_suffix(
        f".{label_format.lower()}", labels_dir, recurse=recurse
    )
    if not label_paths:
        raise FileNotFoundError(
            f"No label files with format '{label_format}' found in '{labels_dir}'"
        )
    label_paths = list(set(label_paths))

    image_paths = []
    if not isinstance(image_format, list):
        image_format = [image_format]
    for suffix in image_format:
        image_paths.extend(
            collect_files_with_suffix(f".{suffix.lower()}", images_dir, recurse=recurse)
        )

    if not image_paths:
        raise FileNotFoundError(
            f"No image files with format(s) '{image_format}' found in '{images_dir}'"
        )
    image_paths = list(set(image_paths))

    if shuffle:
        rand = random.Random(shuffle_seed)
        rand.shuffle(label_paths)

    image_stem_path_map = {path.stem: path for path in image_paths}
    unmatched_labels = []

    all_paths = []
    for label_path in label_paths:
        image_path = image_stem_path_map.get(label_path.stem)
        if image_path:
            if stratify_split and class_data is None:
                raise ArgumentError(
                    "`class_data` is required when `stratify_split` is True. Format contain 'image_stem': 'class_key' key:value pairs"
                )
            class_value = class_data.get(image_path.name) if stratify_split else None
            all_paths.append(
                {"image": image_path, "label": label_path, "class_id": class_value}
            )
        else:
            unmatched_labels.append(label_path)

    if unmatched_labels:
        print(f"Unmatched labels found: {len(unmatched_labels)}")
        for path in unmatched_labels:
            print(f"Unmatched label: {path}", file=sys.stderr)

    match (mode):
        case DatasetSplitMode.All | "all":
            train_data, val_data = split_dataset(
                all_paths,
                split=train_split,
                stratify=stratify_split,
                class_key="class_id" if stratify_split else None,
            )
        case DatasetSplitMode.Collection | "collection":
            train_data, val_data = split_dataset_by_collection(
                all_paths, split=train_split
            )
        case _:
            raise ArgumentError(
                f'Invalid mode argument "{mode}". Options are: "all", "collection".'
            )

    test_data = ([], [])
    if len(val_data[0]) > 0 and test_split > 0:
        split_paths = []
        val_images = [Path(image.filepath) for image in val_data[0]]
        for group in all_paths:
            if group["image"] in val_images:
                split_paths.append(group)

        val_data, test_data = split_dataset(
            split_paths,
            split=1 - (test_split / (1 - train_split)),
            stratify=stratify_split,
            class_key="class_id" if stratify_split else None,
        )

    return train_data, val_data, test_data


# def make_dataset(
#     images_dir: StrPathLike,
#     labels_dir: StrPathLike,
#     *,
#     image_format: str | list[str] = ["jpg", "png"],
#     label_format: str = "txt",
#     train_split: float = 0.75,
#     test_split: float = 0.1,
#     stratify_split: bool = True,
#     mode: DatasetSplitMode | str = DatasetSplitMode.All,  # Addtional args: 'collection'
#     shuffle: bool = True,
#     shuffle_seed: int | None = None,
#     recurse: bool = True,
#     **kwargs,
# ) -> tuple[
#     tuple[list[ImageData], list[AnnotatedLabel]],  # Train
#     tuple[list[ImageData], list[AnnotatedLabel]],  # Validation
#     tuple[list[ImageData], list[AnnotatedLabel]],  # Test
# ]:
#     if kwargs.get("label_format"):
#         label_format = kwargs.pop("label_format")
#     if kwargs.get("image_format"):
#         image_format = kwargs.pop("image_format")
#
#     label_paths = collect_files_with_suffix(
#         f".{label_format.lower()}", labels_dir, recurse=recurse
#     )
#     if not label_paths:
#         raise FileNotFoundError(
#             f"No label files with format '{label_format}' found in '{labels_dir}'"
#         )
#
#     image_paths = []
#     if not isinstance(image_format, list):
#         image_format = [image_format]
#     for suffix in image_format:
#         image_paths.extend(
#             collect_files_with_suffix(f".{suffix.lower()}", images_dir, recurse=recurse)
#         )
#
#     if not image_paths:
#         raise FileNotFoundError(
#             f"No image files with format(s) '{image_format}' found in '{images_dir}'"
#         )
#
#     if shuffle:
#         rand = random.Random(shuffle_seed)
#         rand.shuffle(label_paths)
#
#     image_path_map = {path.stem: path for path in image_paths}
#     unmatched_labels = []
#
#     all_paths = []
#     for label_path in label_paths:
#         image_path = image_path_map.get(label_path.stem)
#         if image_path:
#             all_paths.append({"image": image_path, "label": label_path})
#         else:
#             unmatched_labels.append(label_path)
#
#     if unmatched_labels:
#         print(f"Unmatched labels found: {len(unmatched_labels)}")
#         for path in unmatched_labels:
#             print(f"Unmatched label: {path}", file=sys.stderr)
#
#     match (mode):
#         case DatasetSplitMode.All | "all":
#             train_data, val_data = split_dataset(
#                 all_paths, split=train_split, stratify=stratify_split
#             )
#         case DatasetSplitMode.Collection | "collection":
#             train_data, val_data = split_dataset_by_collection(
#                 all_paths, split=train_split
#             )
#         case _:
#             raise ArgumentError(
#                 f'Invalid mode argument "{mode}". Options are: "all", "collection".'
#             )
#
#     test_data = ([], [])
#     if len(val_data[0]) > 0 and test_split > 0:
#         val_data, test_data = split_dataset(
#             [
#                 {"image": Path(image.filepath), "label": Path(label.filepath)}
#                 for image, label in zip(val_data[0], val_data[1])
#             ],
#             split=1 - (test_split / (1 - train_split)),
#             stratify=stratify_split,
#             class_key=
#         )
#
#     return train_data, val_data, test_data


def overlay_mask(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def split_dataset_by_collection(
    image_label_paths: list[dict[str, StrPathLike]], split=0.75
) -> tuple[
    tuple[list[ImageData], list[AnnotatedLabel]],
    tuple[list[ImageData], list[AnnotatedLabel]],
]:
    collections = dict()
    for data in image_label_paths:
        name = Path(data.get("image")).parent.name
        if name not in collections.keys():
            collections[name] = []
        collections[name].append(data)

    train_ds = ([], [])
    val_ds = ([], [])
    for data in collections.values():
        train, val = split_dataset(data, split)
        train_ds[0].extend(train[0])
        train_ds[1].extend(train[1])
        val_ds[0].extend(val[0])
        val_ds[1].extend(val[1])

    return train_ds, val_ds


def split_dataset(
    image_label_paths: list[dict[str, StrPathLike | None]],
    split: float = 0.75,
    stratify: bool = False,
    class_key: str | None = None,
) -> tuple[
    tuple[list[ImageData], list[AnnotatedLabel]],
    tuple[list[ImageData], list[AnnotatedLabel]],
]:
    """
    Split a dataset of image-label paths into training and validation sets.

    Parameters:
        image_label_paths (list): List of dictionaries with "image", "label", and optional class keys.
        split (float): Fraction of data to use for training (0 to 1).
        stratify (bool): Whether to maintain class proportions in the split.
        class_key (str): Key used to access class grouping information in `image_label_paths`.

    Returns:
        tuple: ((train_images, train_labels), (val_images, val_labels))
    """
    if split <= 0:
        raise ArgumentError("`split` must be positive")
    if stratify and not class_key:
        raise ArgumentError("`class_key` must be provided when stratify is True")

    stems = set()

    def process_subset(
        data_subset: list[dict[str, StrPathLike]], stems: set
    ) -> tuple[list[ImageData], list[AnnotatedLabel]]:
        images = []
        labels = []
        for data in data_subset:
            image_path = Path(data["image"])
            if image_path.stem in stems:
                continue
            stems.add(image_path.stem)
            image = ImageData(image_path)
            images.append(image)
            labels.extend(AnnotatedLabel.from_file(data["label"], image_path.name))
        return images, labels

    if stratify:
        class_groups = dict()

        for data in image_label_paths:
            class_value = data.get(class_key)
            if class_value is None:
                raise ValueError(f"Missing `class_key` '{class_key}' in data: {data}")
            if not class_groups.get(class_value):
                class_groups[class_value] = []
            class_groups[class_value].append(data)

        a_data, b_data = [], []
        for class_data in class_groups.values():
            split_index = int(len(class_data) * split)
            a_data.extend(class_data[:split_index])
            b_data.extend(class_data[split_index:])
    else:
        split_index = int(len(image_label_paths) * split)
        a_data = image_label_paths[:split_index]
        b_data = image_label_paths[split_index:]

    # Make sure there are no duplicates in either data
    remove_alternating_duplicates(a_data, b_data, "image")

    a_images, a_labels = process_subset(a_data, stems)
    b_images, b_labels = process_subset(b_data, stems)

    remove_alternating_duplicate_images_and_labels(
        a_images, a_labels, b_images, b_labels
    )

    return (a_images, a_labels), (b_images, b_labels)


def remove_alternating_duplicates(
    a_data: list[dict], b_data: list[dict], key: str
) -> None:
    """
    Removes duplicates alternately from a_data and b_data based on a specified key.

    Args:
        a_data (list[dict]): The first list of dictionaries.
        b_data (list[dict]): The second list of dictionaries.
        key (str): The key to compare values for duplicates.

    Returns:
        None: Modifies the lists in place.
    """
    dups = 0

    for i in range(
        len(b_data) - 1, -1, -1
    ):  # Iterate in reverse to safely remove items
        b_value = b_data[i][key]
        for i, a in enumerate(a_data):
            if a[key] == b_value:
                if dups % 2 == 0:
                    a_data.pop(i)
                else:
                    b_data.pop(i)
                dups += 1
                break


def remove_alternating_duplicate_images_and_labels(
    a_images: list[ImageData],
    a_labels: list[AnnotatedLabel],
    b_images: list[ImageData],
    b_labels: list[AnnotatedLabel],
):
    seen_images = set()
    toggle = True

    def remove_labels_from_list(labels, filename):
        for j in range(len(labels) - 1, -1, -1):
            if labels[j].image_filename == filename:
                del labels[j]

    for image in a_images:
        seen_images.add(image.filename)

    for i in range(len(b_images) - 1, -1, -1):
        filename = b_images[i].filename
        if filename in seen_images:
            if toggle:
                del b_images[i]
                remove_labels_from_list(b_labels, filename)
            else:
                for j in range(len(a_images) - 1, -1, -1):
                    if a_images[j].filename == filename:
                        del a_images[j]
                        break
                remove_labels_from_list(a_labels, filename)
            toggle = not toggle

    return a_images, a_labels, b_images, b_labels


# def split_dataset(
#     image_label_paths: list[dict[str, StrPathLike]],
#     split: float = 0.75,
#     stratify: bool = False,
# ) -> tuple[
#     tuple[list[ImageData], list[AnnotatedLabel]],
#     tuple[list[ImageData], list[AnnotatedLabel]],
# ]:
#     split_index = int(len(image_label_paths) * split)
#     stems = set()
#
#     def process_subset(
#         data_subset: list[dict[str, StrPathLike]], stems: set
#     ) -> tuple[list[ImageData], list[AnnotatedLabel]]:
#         images = []
#         labels = []
#         foremove_alternating_duplicatesin data_subset:
#             image = ImageData(data["image"])
#             image_stem = Path(image.filename).stem
#             if image_stem in stems:
#                 continue
#             stems.add(image_stem)
#             images.append(image)
#             labels.extend(AnnotatedLabel.from_file(data["label"], image.filename))
#         return images, labels
#
#     a_images, a_labels = process_subset(image_label_paths[:split_index], stems)
#     b_images, b_labels = process_subset(image_label_paths[split_index:], stems)
#
#     return (a_images, a_labels), (b_images, b_labels)


def split_dataset_by_k_fold(k):
    # return generator
    pass


def display_image_and_annotations(
    dataset, idx, save_dir=None, show=True, include_background=False, verbose=True
):
    # Get the image and target from the dataset at index `idx`
    img, target = dataset[idx]
    if len(target["masks"]) == 0 and not include_background:
        return

    img_filepath = Path(dataset.image_paths[idx])

    # Convert image to NumPy array
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).numpy()  # Convert [C, H, W] to [H, W, C]
    else:
        # If it's a PIL Image, convert to NumPy directly
        img_np = np.array(img)

    # Get image dimensions
    height, width = img_np.shape[:2]

    # Create a plot with two subplots for side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Set the title of the plot
    fig.suptitle(img_filepath.name)

    # --- First subplot: Original Image ---
    ax1.imshow(img_np)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # --- Second subplot: Image with Annotations ---
    ax2.imshow(img_np)

    # Get the bounding boxes (convert tensor to NumPy array)
    boxes = target["boxes"].numpy()

    # Get the labels (convert tensor to NumPy array)
    labels = target["labels"].numpy()

    # Get the class names from the dataset
    class_names = dataset.classes

    # Add bounding boxes and class names to the plot
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax2.add_patch(rect)

        # Get the class name
        class_name = class_names[label]

        # Add class name text above the bounding box
        ax2.text(
            x1,
            y1 - 10,  # Slightly above the bounding box
            class_name,
            fontsize=12,
            color="red",
            bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="none", pad=2),
        )

    # Get the masks (ensure masks are tensors and convert to NumPy array)
    masks = target["masks"].numpy()

    # Overlay each mask on the image with transparency
    for mask in masks:
        ax2.imshow(np.ma.masked_where(mask == 0, mask), cmap="jet", alpha=0.5)

    # Set titles and axis labels for the annotated image
    ax2.set_title("Annotated Image")
    ax2.set_xlabel(f"Width (pixels): {width}")
    ax2.set_ylabel(f"Height (pixels): {height}")

    # Add axis ticks
    ax2.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax2.set_yticks(np.arange(0, height, max(1, height // 10)))

    # Hide axis lines for both images
    ax1.axis("off")
    ax2.axis("off")

    # Save the figure containing both subplots (side-by-side comparison)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir).resolve()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        save_filepath = save_dir / f"{img_filepath.stem}_sbs.png"

        plt.savefig(save_filepath)

        if verbose:
            print(f"Image {save_filepath.name} saved to {save_dir}")

    if show:
        # Show the side-by-side images (optional)
        plt.show()

    # Close the plot to free memory
    plt.close()


def display_ground_truth_and_predicted_images(
    dataset,
    idx,
    predicted_images,
    save_dir=None,
    show=True,
    include_background=False,
    verbose=True,
    font_size=12,
    line_width=2,
    text_offset=0,
):
    # Get the image and target from the dataset at index `idx`
    gt_image, target = dataset[idx]
    if len(target["masks"]) == 0 and not include_background:
        return

    gt_filepath = Path(dataset.image_paths[idx])
    pred_filepath = None
    for path in pathify(predicted_images):
        if path.stem == gt_filepath.stem:
            pred_filepath = path
    if pred_filepath is None:
        raise FileNotFoundError(f"Cannot find predicted image '{gt_filepath.name}'")

    # Convert image to NumPy array
    if isinstance(gt_image, torch.Tensor):
        img_np = gt_image.permute(1, 2, 0).numpy()  # Convert [C, H, W] to [H, W, C]
    else:
        # If it's a PIL Image, convert to NumPy directly
        img_np = np.array(gt_image)

    pred_image = Image.open(pred_filepath)
    pred_np = np.array(pred_image)

    # Get image dimensions
    height, width = img_np.shape[:2]

    # Create a plot with two subplots for side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Set the title of the plot
    fig.suptitle(gt_filepath.name)

    # --- First subplot: Ground truth---
    ax1.imshow(img_np)

    # Get the bounding boxes (convert tensor to NumPy array)
    boxes = target["boxes"].numpy()

    # Get the labels (convert tensor to NumPy array)
    labels = target["labels"].numpy()

    # Get the class names from the dataset
    class_names = dataset.classes

    # Add bounding boxes and class names to the plot
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=line_width, edgecolor="red", facecolor="none"
        )
        ax1.add_patch(rect)

        # Get the class name
        class_name = class_names[label]

        # Add class name text above the bounding box
        ax1.text(
            x1,
            y1 - text_offset,  # Slightly above the bounding box
            class_name,
            fontsize=font_size,
            color="red",
            bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="none", pad=2),
        )

    # Get the masks (ensure masks are tensors and convert to NumPy array)
    masks = target["masks"].numpy()

    # Overlay each mask on the image with transparency
    for mask in masks:
        ax1.imshow(np.ma.masked_where(mask == 0, mask), cmap="jet", alpha=0.5)

    # Set titles and axis labels for the annotated image
    ax1.set_title("Ground Truth")
    ax1.set_xlabel(f"Width (pixels): {width}")
    ax1.set_ylabel(f"Height (pixels): {height}")

    # Add axis ticks
    ax1.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax1.set_yticks(np.arange(0, height, max(1, height // 10)))

    # Hide axis lines for both images
    ax1.axis("off")

    # --- Second subplot: Predicted image---
    ax2.imshow(pred_np)
    ax2.set_title("Predicted")
    ax2.axis("off")

    # Save the figure containing both subplots (side-by-side comparison)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir).resolve()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        save_filepath = save_dir / f"{gt_filepath.stem}_sbs.png"

        plt.savefig(save_filepath)

        if verbose:
            print(f"Image {save_filepath.name} saved to {save_dir}")

    if show:
        # Show the side-by-side images (optional)
        plt.show()

    # Close the plot to free memory
    plt.close()


def maskrcnn_get_transform(
    train: bool,
    imgsz=None,
    *,
    augment=False,
    flip_h: float = 0.2,
    flip_v: float = 0.5,
    rot_deg=(180,),
    blur_kernel=(5, 9),
    blur_sigma=(0.1, 5),
):
    transforms = []
    if train:
        # if imgsz is None:
        #     raise ValueError("Missing required argument 'imgsz' for training")
        if augment:
            transforms.append(T.RandomHorizontalFlip(flip_h))
            transforms.append(T.RandomVerticalFlip(flip_v))
            transforms.append(T.RandomRotation(rot_deg))
            transforms.append(T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma))

        if imgsz is not None:
            transforms.append(T.Resize(imgsz))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def plot_paired_images(paired_images, nrows=1, ncols=2, figsize=(15, 15)):
    assert ncols % 2 == 0, "Number of columns must be a multiple of 2"
    assert (
        len(paired_images) >= nrows * ncols
    ), "Number of pairs is less than the dimensions given"

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if col % 2 == 0:
                axes[row][col].set_title(f"{count}: Ground Truth")
            else:
                axes[row][col].set_title(f"{count}: Predicted")

            axes[row][col].imshow(paired_images[count][col % 2])
            axes[row][col].axis("off")
            count += 1

    plt.tight_layout()
    plt.show()
