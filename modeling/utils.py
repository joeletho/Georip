import glob
import json
import os
import random
import shutil
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import ArgumentError
from pathlib import Path
from time import sleep, time
from types import FunctionType
from xml.etree import ElementTree as ET

import cv2
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import PIL
import rasterio
import skimage
import supervision as sv
import torch
import yaml
from matplotlib import pyplot as plt
from numpy import NaN
from PIL import Image, ImageDraw
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from tqdm.auto import tqdm, trange

from ftcnn.modeling.maskrcnn import collate_fn
from ftcnn.utils import (Lock, clear_directory, collect_files_with_suffix,
                         get_cpu_count, pathify)

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

TQDM_INTERVAL = 1 / 100

XYPair = tuple[float | int, float | int]
XYInt = tuple[int, int]
ClassMap = dict[str, str]


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
        filepath = Path(filepath).resolve()
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


class YOLODataset(Serializable):
    _lock = Lock()
    data_frame: pd.DataFrame
    class_map: dict[str, int]
    class_distribution: dict[str, int]
    root_path: Path = Path()

    def __init__(
        self,
        labels: list[AnnotatedLabel],
        images: list[ImageData],
        *,
        compile=True,
        num_workers=None,
    ):
        self.labels = labels
        self.images = images

        if compile and len(labels) > 0 and len(images) > 0:
            if num_workers is None:
                num_workers = 1
            self.compile(num_workers)

    def __request_lock__(self):
        while self._lock.is_locked():
            sleep(TQDM_INTERVAL)
        self._lock.lock()
        return self._lock

    def __free_lock__(self):
        self._lock.unlock()

    def get_num_classes(self):
        return len(self.class_map.keys()) if hasattr(self, "class_map") else 0

    def get_class_distribution(self):
        if not hasattr(self, "class_distribution"):
            return "None"
        return json.dumps(
            self.class_distribution, default=lambda o: o.__dict__, indent=2
        )

    def info(self):
        ntrain_images = (
            self.data_frame.loc[self.data_frame["type"] == "train", "filename"]
            .unique()
            .shape[0]
        )
        ntrain_labels = self.data_frame.loc[self.data_frame["type"] == "train"].shape[0]
        nval_images = (
            self.data_frame.loc[self.data_frame["type"] == "val", "filename"]
            .unique()
            .shape[0]
        )
        nval_labels = self.data_frame.loc[self.data_frame["type"] == "val"].shape[0]

        self.__request_lock__()
        print(
            "YOLO Dataset information\n"
            + f"Number of labels: {len(self.labels)}\n"
            + f"Number of images: {len(self.images)}\n"
            + f"Number of classes: {self.get_num_classes()}\n"
            + f"Training data: {ntrain_images} images, {ntrain_labels} labels\n"
            + f"Validation data: {nval_images} images, {nval_labels} labels\n"
        )
        self.__free_lock__()

    def summary(self):
        ntrain_images = (
            self.data_frame.loc[self.data_frame["type"] == "train", "filename"]
            .unique()
            .shape[0]
        )
        ntrain_labels = self.data_frame.loc[self.data_frame["type"] == "train"].shape[0]
        nval_images = (
            self.data_frame.loc[self.data_frame["type"] == "val", "filename"]
            .unique()
            .shape[0]
        )
        nval_labels = self.data_frame.loc[self.data_frame["type"] == "val"].shape[0]

        self.__request_lock__()
        print(
            "YOLO Dataset summary\n"
            + f"Number of labels: {len(self.labels)}\n"
            + f"Number of images: {len(self.images)}\n"
            + f"Number of classes: {self.get_num_classes()}\n"
            + f"Training data: {ntrain_images} images, {ntrain_labels} labels\n"
            + f"Validation data: {nval_images} images, {nval_labels} labels\n"
            + "Class distribution:\n"
            + self.get_class_distribution()
            + "\n\n"
            + f"Data:\n{self.data_frame}\n"
        )
        self.__free_lock__()

    @staticmethod
    def get_mapped_classes(labels: list[AnnotatedLabel]):
        """Maps each classname to a unique id

        Parameters
        __________
        labels: list[AnnotatedLabel]
            the list of annotated labels to be mapped

        Returns
        _______
        dict[str, int]
            a dict where each key is a classname and values are the associated id
        """
        unique_names = set()
        for label in labels:
            if label.class_name not in unique_names:
                unique_names.add(label.class_name)

        classes = {name: id for id, name in zip(range(len(unique_names)), unique_names)}

        has_background = False
        for name in classes.keys():
            if name.lower() == "background":
                has_background = True
                classes[name] = -1
        if has_background:
            count = 0
            for name, id in classes.items():
                if id != -1:
                    classes[name] = count
                    count += 1
        return classes

    @staticmethod
    def normalize_point(x, y, width, height, xoffset, yoffset):
        dw = 1 / float(width)
        dh = 1 / float(height)
        x = float(x)
        y = float(y)
        if xoffset and yoffset:
            xoffset = float(xoffset)
            yoffset = float(yoffset)
            x = (x + xoffset) * dw
            y = (y + yoffset) * dh
            xoffset *= dw
            yoffset *= dh
        else:
            x *= dw
            y *= dh
        return round(x, 6), round(y, 6), round(xoffset, 6), round(yoffset, 6)

    @staticmethod
    def convert_bbox_to_yolo(
        *,
        bbox: BBox,
        imgsize: XYPair,
    ):
        """Converts an non-formatted bounding box to YOLO format
        Modifies the original `bbox` in-place.

        Parameters
        __________
        bbox: BBox
            the dimensions of the bounding box
        imgsize: XYPair
            the width and height of the image

        Returns
        _______
        BBox
            the converted bbox object
        """
        if bbox.x > 1 or bbox.y > 1 or bbox.width > 1 or bbox.height > 1:
            x, y, w, h = YOLODataset.normalize_point(
                bbox.x,
                bbox.y,
                imgsize[0],
                imgsize[1],
                bbox.width,
                bbox.height,
            )
            bbox.x = round(x / 2, 6)
            bbox.y = round(y / 2, 6)
            bbox.width = w
            bbox.height = h
        return bbox

    def compile(self, num_workers=1):
        """Compiles the labels and images into a dataset

        Returns
        _______
        Self
        """
        data = {
            "type": [],
            "class_id": [],
            "class_name": [],
            "bbox_x": [],
            "bbox_y": [],
            "bbox_w": [],
            "bbox_h": [],
            "filename": [],
            "width": [],
            "height": [],
            "path": [],
            "segments": [],
        }
        if not (len(self.labels) and len(self.images)):
            raise ValueError("Dataset does not contain valid data.")

        self.labels = list(set(self.labels))
        self.images = list(set(self.images))

        self.class_map = YOLODataset.get_mapped_classes(self.labels)
        self.class_distribution = {name: 0 for name in self.class_map.keys()}

        indices_to_remove = []

        def __exec__(labels):
            total_updates = len(labels)
            updates = 0
            self.__request_lock__()
            pbar = trange(
                total_updates,
                desc="Compiling YOLODataset labels and images",
                leave=False,
            )
            pbar.refresh()
            self.__free_lock__()

            start = time()
            for i, label in enumerate(labels):
                self.__request_lock__()
                self.class_distribution[label.class_name] += 1
                self.__free_lock__()

                image_data = None
                for image in self.images:
                    if label.image_filename == image.filename:
                        image_data = image
                        break
                if image_data is None:
                    self.__request_lock__()
                    print(
                        f"Image '{label.image_filename}' not found in labels -- label flagged for removal",
                        file=sys.stderr,
                    )
                    self.__free_lock__()
                    indices_to_remove.append(i)
                    continue

                if label.bbox is None:
                    label.bbox = BBox(-1, -1, -1, -1)
                else:
                    YOLODataset.convert_bbox_to_yolo(
                        imgsize=image_data.shape, bbox=label.bbox
                    )

                if label.segments is not None:
                    for seg in label.segments:
                        if float(seg) <= 1:
                            continue
                        # If any one segment is not normalized, assume none are
                        for i, s in enumerate(label.segments):
                            s = float(s)
                            s /= image_data.shape[i % 2]
                            label.segments[i] = s
                        break

                    if label.bbox == BBox(-1, -1, -1, -1):
                        label.bbox = YOLODataset.convert_bbox_to_yolo(
                            imgsize=image_data.shape,
                            bbox=convert_segment_to_bbox(label.segments),
                        )
                else:
                    label.segments = ""

                self.__request_lock__()
                data["type"].append(
                    label.type
                    if label.type is not None and len(label.type) > 0
                    else "None"
                )
                data["class_id"].append(self.class_map.get(str(label.class_name)))
                data["class_name"].append(label.class_name)
                data["filename"].append(image_data.filename)
                data["width"].append(image_data.shape[0])
                data["height"].append(image_data.shape[1])
                data["path"].append(image_data.filepath)
                data["bbox_x"].append(label.bbox.x)
                data["bbox_y"].append(label.bbox.y)
                data["bbox_w"].append(label.bbox.width)
                data["bbox_h"].append(label.bbox.height)
                data["segments"].append(
                    " ".join([str(point) for point in label.segments]),
                )
                self.__free_lock__()

                if time() - start >= TQDM_INTERVAL:
                    self.__request_lock__()
                    pbar.update()
                    updates += 1
                    start = time()
                    self.__free_lock__()
            if updates < total_updates:
                pbar.update(total_updates - updates)
            pbar.close()

        if len(self.labels) < 100:
            __exec__(self.labels)
        else:
            num_workers = max(1, min(num_workers, get_cpu_count()))

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                batch = len(self.labels) // num_workers
                i = 0
                while i < len(self.labels):
                    end = i + batch
                    futures.append(executor.submit(__exec__, self.labels[i:end]))
                    i = end
                for future in as_completed(futures):
                    msg = future.exception()
                    if msg is not None:
                        raise RuntimeError(msg)

        self.data_frame = pd.DataFrame.from_dict(data)
        self.data_frame = self.data_frame.drop_duplicates()

        if len(indices_to_remove) > 0:
            self.__request_lock__()
            print("Cleaning unused labels ...")
            for counter, index in enumerate(indices_to_remove):
                pop_index = index - counter + 1
                if pop_index > len(self.labels):
                    break
                label = self.labels.pop(pop_index)
                print(f"  Removed: {label}")
            self.__free_lock__()

        return self

    def to_csv(self, output_path: str | Path, **kwargs):
        """Saves the dataset to a CSV file

        Parameters
        __________
        output_path: str
            the path of the destination file
        kwargs: any
            additional keyword arguments passed to the DataFrame.to_csv function
        """
        self.data_frame.to_csv(output_path, index=False, **kwargs)

    @staticmethod
    def from_csv(path: str | Path, **kwargs):
        """Constructs and returns a YOLODataset from a CSV file.

        The dataset is required to have the following columns in any order:
        -----------------------------------------------------
           class_id: str
               the id of the label class
           class_name: str
               the name of the label class
           bbox_x: float
               the normalized x-coordinate of the bounding box center
           bbox_y: float
               the normalized y-coordinate of the bounding box center
           bbox_w: float
               the normalized width of the bounding box
           bbox_h: float
               the normalized height of the bounding box
           segments: list[float]
               a list of points constructng a polygon
           filename: str
               the filename of the image, including file extension
           path: str
               the path of the image, including filename
           width: float|int
               the width of the image
           height: float|int
               the height of the image

        Parameter
        _________
        path: str
            the path of the destination direcrory, including filename

        Returns
        _______
        a newly constructed YOLODataset object
        """
        compile = kwargs.pop("compile", None)
        compile = compile if compile is not None else True
        num_workers = kwargs.pop("num_workers", None)

        df = pd.read_csv(path)
        image_map = {row["filename"]: str(row["path"]) for _, row in df.iterrows()}
        image_names = set()
        images = []
        labels = parse_labels_from_csv(path, **kwargs)
        total_updates = len(labels)
        updates = 0
        start = time()
        pbar = trange(total_updates, desc="Collecting images", leave=False)
        for label in labels:
            image_name = label.image_filename
            if image_name not in image_names:
                image_names.add(image_name)
                image = ImageData(image_map[image_name])
                images.append(image)
            if time() - start >= TQDM_INTERVAL:
                pbar.update()
                updates += 1
                start = time()
        if updates < total_updates:
            pbar.update(total_updates - updates)
        pbar.close()

        return YOLODataset(
            labels=labels, images=images, compile=compile, num_workers=num_workers
        )

    def generate_label_files(
        self,
        dest_path: str | Path,
        *,
        clear_dir: bool = False,
        overwrite_existing: bool = False,
        use_segments=True,
    ):
        """Generates the label files used by YOLO
        Files are saved in the `dest_path` directory with the basename of their associated image.
        If the image filename is `img_001.png`, the label file will be `img_001.txt`.

        Output format:
        [class id] [bbox x] [bbox y] [bbox width] [bbox height]

        Parameters
        __________
        dest_path: str
            the path to directory in which to save the generated files
        clear_directory: bool
            erase all files in the `dest_path` directory
        overwrite_existing: bool
            overwrite existing files in the `dest_path` directory

        Example
        _______
            # img_001.txt
            6 0.129024 0.3007129669189453 0.0400497777777776 0.045555555555555564
            2 0.08174603174603165 0.22560507936507962 0.13915343915343897 0.1798772486772488
        """

        dest_path = Path(dest_path).resolve()
        if not dest_path.exists():
            dest_path.mkdir(parents=True)

        existing_label_files = glob.glob(str(dest_path / "*.txt"))

        if clear_dir:
            clear_directory(dest_path)

        if not clear_dir and overwrite_existing:
            existing = {Path(path).stem: path for path in existing_label_files}
            for _, row in self.data_frame.iterrows():
                filename = str(row["filename"])
                if existing.get(filename):
                    os.remove(existing[filename])
                    existing[filename] = ""

        annotations = 0
        backgrounds = 0
        files = set()
        total_updates = self.data_frame.shape[0]
        updates = 0
        start = time()
        self.__request_lock__()
        pbar = trange(total_updates, desc="Generating labels")
        pbar.refresh()
        self.__free_lock__()
        for _, row in self.data_frame.iterrows():
            _dest_path: Path = dest_path
            filename = Path(str(row["filename"]))
            _type = str(row.get("type"))
            class_id = row["class_id"]
            is_background = class_id == -1

            points = []
            if not is_background:
                if use_segments:
                    if len(str(row["segments"])) == 0:
                        is_background = True
                    else:
                        points = str(row["segments"]).split()
                        if 1 < len(points) < 3 * 2:
                            raise ValueError(
                                "Segments must contain 3 or more point pairs (x, y)"
                            )
                else:
                    bbox_x = float(row["bbox_x"])
                    bbox_y = float(row["bbox_y"])
                    bbox_w = float(row["bbox_w"])
                    bbox_h = float(row["bbox_h"])
                    points = [bbox_x, bbox_y, bbox_w, bbox_h]
                    for point in points:
                        if point < 0:
                            is_background = True
                            break

            if _type in ["train", "val"]:
                _dest_path = self.root_path / "labels" / _type
            label_path = str(_dest_path / f"{filename.stem}.txt")

            if is_background:
                label_desc = ""
            else:
                label = row["class_name"]
                label_desc = (
                    f"{self.class_map[str(label)]} {' '.join(map(str, points))}\n"
                )

            files.add(label_path)
            with open(label_path, "a+") as f:
                f.write(label_desc)
                if is_background:
                    backgrounds += 1
                else:
                    annotations += 1
            if time() - start < TQDM_INTERVAL:
                pbar.update()
                updates += 1
                start = time()
        if updates < total_updates:
            pbar.update(total_updates - updates)

        self.__request_lock__()
        pbar.set_description("Complete")
        pbar.close()
        print(
            f"Successfully generated {annotations} annoations and {backgrounds} backgrounds to {len(files)} files"
        )
        self.__free_lock__()

    def generate_yaml_file(
        self,
        root_abs_path: str | Path,
        dest_abs_path: str | Path | None = None,
        *,
        filename: str = "data.yaml",
        train_path: str | Path = "images/train",
        val_path: str | Path = "images/val",
    ):
        """Generates and saves the YAML data file used by YOLO

        Parameters
        __________
        root_abs_path: str
            the absolute path of the dataset root directory
        dest_abs_path: str | None
            the absolute path of the output file. If None, the file will
            be saved the root directory as 'data.yaml'.
        train_path: str
            the relative path of training images directory
        val_path: str
            the relative path of validation images directory
        test_path: str
            the relative path of test images directory
        """
        if dest_abs_path is not None:
            if os.path.isfile(dest_abs_path):
                raise Exception(f"{dest_abs_path} is not a valid directory")

        if dest_abs_path is None:
            dest_abs_path = root_abs_path

        root_abs_path = root_abs_path
        dest_abs_path = Path(dest_abs_path, filename).resolve()

        with open(dest_abs_path, "w") as f:
            f.write(f"path: {root_abs_path}\n")
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            f.write("names:\n")
            for name, id in self.class_map.items():
                if name.lower() != "background":
                    f.write(f"  {id}: {name}\n")
        self.root_path = Path(root_abs_path)

    def split_data(
        self,
        images_dir,
        labels_dir,
        *,
        split=0.7,
        shuffle=True,
        recurse=True,
        save=True,
        mode="all",
        background_bias=None,
        **kwargs,
    ):

        train_ds, val_ds = make_dataset(
            images_dir,
            labels_dir,
            mode=mode,
            split=split,
            shuffle=shuffle,
            recurse=recurse,
            **kwargs,
        )

        for train_image in train_ds[0]:
            name = str(train_image.filename)
            self.data_frame.loc[self.data_frame["filename"] == name, "type"] = "train"
            class_name = self.data_frame.loc[
                self.data_frame["filename"] == name, "class_name"
            ].iloc[0]
            class_id = self.data_frame.loc[
                self.data_frame["filename"] == name, "class_id"
            ].iloc[0]
            for label in train_ds[1]:
                if label.image_filename == name:
                    label.class_name = class_name
                    label.class_id = class_id

        for val_image in val_ds[0]:
            name = str(val_image.filename)
            self.data_frame.loc[self.data_frame["filename"] == name, "type"] = "val"
            class_name = self.data_frame.loc[
                self.data_frame["filename"] == name, "class_name"
            ].iloc[0]
            class_id = self.data_frame.loc[
                self.data_frame["filename"] == name, "class_id"
            ].iloc[0]
            for label in val_ds[1]:
                if label.image_filename == name:
                    label.class_name = class_name
                    label.class_id = class_id

        if background_bias is not None:
            ntrain_class = self.data_frame.loc[
                (self.data_frame["type"] == "train")
                & (self.data_frame["class_id"] != -1)
            ].shape[0]
            ntrain_bkgd = self.data_frame.loc[
                (self.data_frame["type"] == "train")
                & (self.data_frame["class_id"] == -1)
            ].shape[0]
            nbkgd_rm = (
                ntrain_bkgd - ntrain_class
                if ntrain_bkgd / ntrain_class > background_bias
                else 0
            )

            if nbkgd_rm > 0:
                background_rows = self.data_frame[
                    (self.data_frame["type"] == "train")
                    & (self.data_frame["class_id"] == -1)
                ]
                rows_to_drop = background_rows.sample(n=nbkgd_rm).index
                self.data_frame = self.data_frame.drop(rows_to_drop)

            nval_class = self.data_frame.loc[
                (self.data_frame["type"] == "val") & (self.data_frame["class_id"] != -1)
            ].shape[0]
            nval_bkgd = self.data_frame.loc[
                (self.data_frame["type"] == "val") & (self.data_frame["class_id"] == -1)
            ].shape[0]
            nbkgd_rm = (
                nval_bkgd - nval_class
                if nval_bkgd / nval_class > background_bias
                else 0
            )

            if nbkgd_rm > 0:
                background_rows = self.data_frame[
                    (self.data_frame["type"] == "val")
                    & (self.data_frame["class_id"] == -1)
                ]
                rows_to_drop = background_rows.sample(n=nbkgd_rm).index
                self.data_frame = self.data_frame.drop(rows_to_drop)

        if save:
            copy_images_and_labels(
                image_paths=[image.filepath for image in train_ds[0]],
                label_paths=[label.filepath for label in train_ds[1]],
                images_dest=self.root_path / "images" / "train",
                labels_dest=self.root_path / "labels" / "train",
            )
            copy_images_and_labels(
                image_paths=[image.filepath for image in val_ds[0]],
                label_paths=[label.filepath for label in val_ds[1]],
                images_dest=self.root_path / "images" / "val",
                labels_dest=self.root_path / "labels" / "val",
            )

        self.data_frame = pd.DataFrame(
            pd.concat(
                [
                    self.data_frame.loc[self.data_frame["type"] == "train"],
                    self.data_frame.loc[self.data_frame["type"] == "val"],
                ],
                ignore_index=True,
            )
        )
        self.data_frame = self.data_frame.sort_values(by="type", ignore_index=True)

        self.images = train_ds[0] + val_ds[0]
        self.labels = train_ds[1] + val_ds[1]

        return train_ds, val_ds


class YOLODatasetLoader(torch.utils.data.Dataset):
    def __init__(
        self, classes, images_dir, annotations_dir, transforms=None, recurse=False
    ):
        """
        Args:
            images_dir (str): Directory where the images are stored.
            annotations_dir (str): Directory where the YOLO annotation files are stored.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        if not isinstance(classes, list):
            classes = [classes]
        self.classes = classes
        self.images_dir = Path(images_dir).resolve()
        self.labels_dir = Path(annotations_dir).resolve()
        self.transforms = transforms

        def get_image_paths(dir, paths, recurse=recurse):
            for path in dir.iterdir():
                if not path.is_file():
                    if recurse:
                        get_image_paths(path, paths, recurse)
                    continue
                if str(path.suffix).lower() in (".png", ".jpg", ".jpeg"):
                    paths.append(path)

        self.image_paths = []
        get_image_paths(self.images_dir, self.image_paths, recurse)
        self.image_paths.sort()

    def __getitem__(self, idx):
        # Get image file name and corresponding annotation file
        image_file = self.image_paths[idx]
        image_path = self.images_dir / image_file
        label_path = self.labels_dir / Path(image_file.stem).with_suffix(".txt")

        # Load the image
        image = read_image(image_path)
        image = tv_tensors.Image(image)

        # Get image dimensions
        img_height, img_width = F.get_size(image)  # returns (H, W)

        # Initialize lists for boxes, labels, masks
        boxes = []
        labels = []
        masks = []

        # Check if annotation file exists
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Skip invalid lines

                    class_id = int(parts[0])

                    if len(parts) == 5:
                        # Standard YOLO format: class_id x_center y_center width height
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert from normalized coordinates to pixel coordinates
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height

                        # Convert from center coordinates to corner coordinates
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2

                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

                        # Create a mask from the bounding box
                        mask = torch.zeros((img_height, img_width), dtype=torch.uint8)
                        x_min_int = int(round(x_min))
                        y_min_int = int(round(y_min))
                        x_max_int = int(round(x_max))
                        y_max_int = int(round(y_max))
                        mask[y_min_int:y_max_int, x_min_int:x_max_int] = 1
                        masks.append(mask.numpy())
                    else:
                        # Assume the rest of the parts are polygon coordinates
                        # Format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
                        coords = list(map(float, parts[1:]))
                        if len(coords) % 2 != 0:
                            continue  # Invalid polygon

                        x_coords = coords[::2]
                        y_coords = coords[1::2]

                        # Convert normalized coordinates to pixel coordinates
                        x_coords = [x * img_width for x in x_coords]
                        y_coords = [y * img_height for y in y_coords]

                        # Create a polygon
                        polygon = [(x, y) for x, y in zip(x_coords, y_coords)]

                        # Create a mask from the polygon
                        mask_img = Image.new("L", (img_width, img_height), 0)
                        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
                        mask = np.array(mask_img, dtype=np.uint8)
                        masks.append(mask)

                        # Compute bounding box
                        x_min = min(x_coords)
                        x_max = max(x_coords)
                        y_min = min(y_coords)
                        y_max = max(y_coords)
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

        else:
            # If annotation file doesn't exist, return empty annotations
            boxes = []
            labels = []
            masks = []

        # Convert to tensors
        if len(boxes) > 0:
            boxes = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(image)
            )
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = tv_tensors.Mask(masks)
            # Compute area
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), format="XYXY", canvas_size=F.get_size(image)
            )
            labels = torch.empty(0, dtype=torch.int64)
            masks = tv_tensors.Mask(torch.zeros((0, *F.get_size(image))))
            area = torch.empty(0, dtype=torch.float32)
            iscrowd = torch.empty(0, dtype=torch.int64)

        image_id = idx

        # Prepare the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_paths)

    def make_datasets(
        self,
        batch_train,
        batch_val,
        split_ratio=0.75,
        shuffle_train=True,
        shuffle_val=False,
    ):
        # Calculate split sizes for 70% train, 20% validation, and 10% test
        train_split_ratio = split_ratio
        val_split_ratio = 1 - split_ratio

        train_size = int(len(self) * train_split_ratio)
        val_size = int(len(self) * val_split_ratio)

        # Generate random indices for splitting
        indices = torch.randperm(len(self)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]

        # Create training, validation, and test subsets
        train_dataset = torch.utils.data.Subset(self, train_indices)
        dataset_val = torch.utils.data.Subset(self, val_indices)

        # Define the training data loader using the subset
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_train,
            shuffle=shuffle_train,
            collate_fn=maskrcnn.collate_fn,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_val,
            shuffle=shuffle_val,
            collate_fn=collate_fn,
        )

        return data_loader_train, data_loader_val

    @staticmethod
    def get_data_loaders_from_yaml(
        data_yaml,
        batch_train,
        batch_val,
        imgsz=None,
        shuffle_train=False,
        shuffle_val=False,
        transform=True,
        train=True,
        **transform_kwargs,
    ):
        with open(data_yaml, "r") as file:
            data = yaml.safe_load(file)

        root_path = Path(data.get("path", ""))
        train_path = Path(data.get("train", ""))
        val_path = Path(data.get("val", ""))
        names = data.get("names", {})
        classes = list(names.values())

        train_images_path = root_path / train_path
        train_labels_path = (
            root_path / "labels" / "/".join(str(part) for part in train_path.parts[1:])
        )
        val_images_path = root_path / val_path
        val_labels_path = (
            root_path / "labels" / "/".join(str(part) for part in val_path.parts[1:])
        )

        ds_train = YOLODatasetLoader(
            classes,
            train_images_path,
            train_labels_path,
            transforms=(
                maskrcnn_get_transform(train=train, imgsz=imgsz, **transform_kwargs)
                if transform
                else None
            ),
        )
        ds_val = YOLODatasetLoader(
            classes,
            val_images_path,
            val_labels_path,
            transforms=(
                maskrcnn_get_transform(train=train, imgsz=imgsz, **transform_kwargs)
                if transform
                else None
            ),
        )
        loader_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=batch_train,
            shuffle=shuffle_train,
            collate_fn=collate_fn,
        )
        loader_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=batch_val,
            shuffle=shuffle_val,
            collate_fn=collate_fn,
        )

        return classes, (loader_train, loader_val), (ds_train, ds_val)
        #
        # train = {
        #     "images": [],
        #     "labels": [],
        # }
        #
        # val = {
        #     "images": [],
        #     "labels": [],
        # }
        #
        # # Get all images
        # for i, parent in enumerate([train_images_path, val_images_path]):
        #     for path in parent.iterdir():
        #         if not path.is_file():
        #             continue
        #         if i % 2:
        #             val["images"].append(path)
        #         else:
        #             train["images"].append(path)
        #
        # # Match image paths to labels
        # for type in ["train", "val"]:
        #     data = None
        #     label_paths = None
        #     if type == "train":
        #         data = train
        #         label_paths = [
        #             path
        #             for path in train_labels_path.iterdir()
        #             if path.suffix == ".txt"
        #         ]
        #     else:
        #         data = val
        #         label_paths = [
        #             path for path in val_labels_path.iterdir() if path.suffix == ".txt"
        #         ]
        #
        #     for i, img_path in enumerate(data["images"]):
        #         found = False
        #         for lbl_path in label_paths:
        #             if lbl_path.stem == img_path.stem:
        #                 data["labels"].append(str(lbl_path))
        #                 data["images"][i] = str(img_path)
        #                 found = True
        #         if not found:
        #             raise FileNotFoundError(
        #                 f"Label path for image '{img_path.name}' not found in '{train_path if type == 'train' else val_path}'"
        #             )
        #
        # dl_args = {
        #     0: {
        #         "batch": batch_train,
        #         "shuffle": shuffle_train,
        #     },
        #     1: {
        #         "batch": batch_val,
        #         "shuffle": shuffle_val,
        #     },
        # }
        #
        # # https://github.com/pytorch/vision/blob/main/references/detection/utils.py
        # def collate_fn(batch):
        #     return tuple(zip(*batch))
        #
        # data_loaders = []
        # for i, data in enumerate([train, val]):
        #     ds = TensorDataset(
        #         torch.as_tensor(data["images"]), torch.as_tensor(data["labels"])
        #     )
        #     data_loaders.append(
        #         torch.utils.data.DataLoader(
        #             ds,
        #             batch_size=dl_args[i]["batch"],
        #             shuffle=dl_args[i]["shuffle"],
        #             collate_fn=collate_fn,
        #         )
        #     )

        # return classes, ds_train, ds_val

    def draw(
        self,
        show=True,
        save_dir=None,
        include_background=False,
        verbose=False,
        pbar=False,
        leave=False,
        pbar_desc="Drawing annotations",
    ):
        if pbar:
            total_updates = len(self)
            updates = 0
            start = time()
            pbar = trange(total_updates, desc=pbar_desc, leave=leave)
        for i in range(len(self)):
            display_image_and_annotations(
                self,
                idx=i,
                save_dir=save_dir,
                show=show,
                include_background=include_background,
                verbose=verbose,
            )
            if pbar and time() - start >= TQDM_INTERVAL:
                pbar.update()
                updates += 1
                start = time()
        if pbar:
            if updates < total_updates:
                pbar.update(total_updates - updates)
            pbar.close()


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


def remove_rows(df, pred_key: FunctionType):
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
                raise ValueError(f"Row index {i}: key '{class_id_key}' not found")
            if data.get("class_name") is None:
                raise ValueError(f"Row index {i}: key '{class_name_key}' not found")
            if data.get("filename") is None:
                raise ValueError(f"Row index {i}: key '{image_filename_key}' not found")

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
    hits = 0
    for i in range(len(points) - 1):
        if points[i] == points[i + 1]:
            if hits > 1:
                points = list(set(points))
                break
            hits += 1

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
    images_dir,
    labels_dir,
    *,
    image_format=["jpg", "png"],
    label_format="txt",
    split=0.75,
    mode="all",  # Addtional args: 'collection'
    shuffle=True,
    recurse=True,
    **kwargs,
):
    if kwargs.get("label_format"):
        label_format = kwargs.pop("label_format")
    if kwargs.get("image_format"):
        image_format = kwargs.pop("image_format")

    label_paths = collect_files_with_suffix(
        f".{label_format.lower()}", labels_dir, recurse=recurse
    )
    image_paths = []
    if not isinstance(image_format, list):
        image_format = [image_format]
    for suffix in image_format:
        image_paths.extend(
            collect_files_with_suffix(f".{suffix.lower()}", images_dir, recurse=recurse)
        )

    if shuffle:
        random.shuffle(label_paths)

    label_path_map = {path.stem: path for path in label_paths}

    all_paths = []
    for stem, label_path in label_path_map.items():
        found = False
        for image_path in image_paths:
            if image_path.stem == stem:
                all_paths.append(
                    {
                        "image": image_path,
                        "label": label_path,
                    }
                )
                found = True
                break
        if not found:
            print(f"Label for '{stem}' not found", file=sys.stderr)
            label_paths.remove(label_path)
            label_path.unlink()

    match (mode):
        case "all":
            train_data, val_data = split_dataset(all_paths, split=split)
        case "collection":
            train_data, val_data = split_dataset_by_collection(all_paths, split=split)
        case _:
            raise ArgumentError(
                f'Invalid mode argument "{mode}". Options are: "all", "collection".'
            )

    return train_data, val_data


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


def yolo_create_dataset_from_dataframe(df, *, imgdir, parallel=True):
    pbar = trange(3, position=0, desc="Building YOLO Dataset")

    labels = parse_labels_from_dataframe(df)
    pbar.update()

    images = parse_images_from_labels(labels, imgdir, parallel=parallel)
    pbar.update()

    ds = YOLODataset(labels=labels, images=images)
    pbar.update()

    pbar.set_description("Complete")
    pbar.close()

    return ds


def split_dataset_by_collection(image_label_paths, split=0.75):
    collections = dict()
    for data in image_label_paths:
        name = data.get("image").parent.name
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


def split_dataset(image_label_paths, split=0.75):
    split = int(len(image_label_paths) * split)
    train_images = []
    train_labels = []
    stems = set()
    for data in image_label_paths[:split]:
        image = ImageData(data["image"])
        image_stem = Path(image.filename).stem
        if image_stem in stems:
            continue
        stems.add(image_stem)
        train_images.append(image)
        train_labels.extend(AnnotatedLabel.from_file(data["label"], image.filename))

    val_images = []
    val_labels = []
    for data in image_label_paths[split:]:
        image = ImageData(data["image"])
        image_stem = Path(image.filename).stem
        if image_stem in stems:
            continue
        stems.add(image_stem)
        val_images.append(image)
        val_labels.extend(AnnotatedLabel.from_file(data["label"], image.filename))

    return (train_images, train_labels), (val_images, val_labels)


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
):
    # Get the image and target from the dataset at index `idx`
    gt_image, target = dataset[idx]
    if len(target["masks"]) == 0 and not include_background:
        return

    gt_filepath = Path(dataset.image_paths[idx])
    pred_filepath = None
    for path in pathify(predicted_images):
        if path.name == gt_filepath.name:
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
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax1.add_patch(rect)

        # Get the class name
        class_name = class_names[label]

        # Add class name text above the bounding box
        ax1.text(
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
