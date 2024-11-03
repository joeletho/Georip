import glob
import json
import os
import random
import shutil
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree as ET

import cv2
import matplotlib as plt
import numpy as np
import pandas as pd
import rasterio
import skimage
import supervision as sv
from numpy import NaN
from PIL import Image
from tqdm.auto import tqdm, trange

from ftcnn.utils import (clear_directory, collect_files_with_suffix,
                         get_cpu_count)

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)


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
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
        )

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
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.filepath))

    def getPixelAtCoords(self, x, y, **kwargs):
        if self.extension != ".tif":
            raise TypeError(
                f"Image type '{self.extension}' does not contain georeference data"
            )
        image = rasterio.open(self.filepath)
        data = image.index(x, y, **kwargs)
        image.close()
        return data

    def getCoordsAtPixel(self, x, y, **kwargs):
        if self.extension != ".tif":
            raise TypeError(
                f"Image type '{self.extension}' does not contain georeference data"
            )
        image = rasterio.open(self.filepath)
        data = image.xy(x, y, **kwargs)
        image.close()
        return data


class AnnotatedLabel(Serializable):
    def __init__(
        self,
        *,
        class_id: int | None = None,
        class_name: str,
        bbox: BBox | None = None,
        segments: list[float] | None = None,
        image_filename: str,
    ):
        self.class_id: int | None = class_id
        self.class_name: str = class_name
        self.bbox: BBox | None = bbox
        self.image_filename: str = image_filename
        self.segments: list[float] | None = segments

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(
            (
                self.class_id,
                self.class_name,
                self.bbox,
                str(self.segments),
                self.image_filename,
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
    def from_file(filepath):
        filepath = Path(filepath).resolve()
        annotations = []
        with open(filepath) as f:
            for line in f:
                annotations.append(AnnotatedLabel.parse_label(line))
        return annotations


class YOLODataset(Serializable):
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

        print(
            "YOLO Dataset information\n"
            + f"Number of labels: {len(self.labels)}\n"
            + f"Number of images: {len(self.images)}\n"
            + f"Number of classes: {self.get_num_classes()}\n"
            + f"Training data: {ntrain_images} images, {ntrain_labels} labels\n"
            + f"Validation data: {nval_images} images, {nval_labels} labels\n"
        )

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
        unique_names = []
        for label in labels:
            if label.class_name not in unique_names:
                unique_names.append(label.class_name)

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

        self.labels = list(set(self.labels))
        self.images = list(set(self.images))
        self.class_map = YOLODataset.get_mapped_classes(self.labels)
        self.class_distribution = {name: 0 for name in self.class_map.keys()}

        indices_to_remove = []

        def __exec__(labels):
            for i, label in enumerate(
                tqdm(
                    labels, desc="Compiling YOLODataset labels and images", leave=False
                )
            ):
                self.class_distribution[label.class_name] += 1

                image_data = None
                for image in self.images:
                    if label.image_filename == image.filename:
                        image_data = image
                        break
                if image_data is None:
                    print(
                        f"Image '{label.image_filename}' not found in labels -- label flagged for removal",
                        file=sys.stderr,
                    )
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

                data["type"].append("None")
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

        if len(indices_to_remove) > 0:
            print("Cleaning unused labels ...")
            for counter, index in enumerate(indices_to_remove):
                pop_index = index - counter + 1
                if pop_index > len(self.labels):
                    break
                label = self.labels.pop(pop_index)
                print(f"  Removed: {label}")

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
        for label in tqdm(labels, desc="Collecting images", leave=False):
            image_name = label.image_filename
            if image_name not in image_names:
                image_names.add(image_name)
                image = ImageData(image_map[image_name])
                images.append(image)

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
        _dest_path: Path = Path(dest_path)
        if not _dest_path.exists():
            _dest_path.mkdir(parents=True)

        existing_label_files = glob.glob(str(Path(_dest_path, "*.txt").resolve()))

        if clear_dir:
            clear_directory(dest_path)

        print("  Classes:", self.class_map)

        if not clear_dir and overwrite_existing:
            existing = {Path(path).stem: path for path in existing_label_files}
            for _, row in self.data_frame.iterrows():
                filename = str(row["filename"])
                if existing.get(filename):
                    os.remove(existing[filename])
                    existing[filename] = ""

        nlabels = 0
        files = set()
        pbar = tqdm(total=self.data_frame.shape[0], desc="Generating labels")
        for _, row in self.data_frame.iterrows():
            filename = str(row["filename"])
            is_background = int(row["class_id"]) == -1

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

            label_path = str(Path(_dest_path, f"{Path(filename).stem}.txt").resolve())

            if is_background:
                label_desc = ""
            else:
                label = row["class_name"]
                files.add(label_path)
                label_desc = (
                    f"{self.class_map[str(label)]} {' '.join(map(str, points))}\n"
                )

            with open(label_path, "a+") as f:
                if label_desc not in f:
                    f.write(label_desc)
                nlabels += 1
            pbar.update()
        pbar.close()
        print(
            f"Successfully generated {nlabels} labels to {len(files)} files in {_dest_path}"
        )

    def generate_yaml_file(
        self,
        root_abs_path: str | Path,
        dest_abs_path: str | Path | None = None,
        *,
        filename: str = "data.yaml",
        train_path: str | Path = "images/train",
        val_path: str | Path = "images/val",
        test_path: str | Path = "images/test",
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

        root_abs_path = Path(root_abs_path).resolve()
        dest_abs_path = Path(dest_abs_path, filename).resolve()
        data_paths = {
            "train": Path(train_path).resolve(),
            "val": Path(val_path).resolve(),
            "test": Path(test_path).resolve(),
        }

        for kind, path in data_paths.items():
            data_paths[kind] = path.relative_to(root_abs_path)

        print(f"Generating {dest_abs_path}")

        with open(dest_abs_path, "w") as f:
            f.write(f"path: {root_abs_path}\n")
            f.write(f"train: {data_paths.get('train')}\n")
            f.write(f"val: {data_paths.get('val')}\n")
            f.write(f"test: {data_paths.get('test')}\n\n")
            f.write("names:\n")
            for name, id in self.class_map.items():
                if name.lower() != "background":
                    f.write(f"  {id}: {name}\n")
        self.root_path = Path(root_abs_path)
        print(f"File saved successfully to {dest_abs_path}")

    def split_data(
        self,
        images_dir,
        labels_dir,
        *,
        split=0.7,
        shuffle=True,
        recurse=True,
        save=True,
        background_bias=None,
    ):
        data = make_dataset(
            images_dir, labels_dir, split=split, shuffle=shuffle, recurse=recurse
        )
        for image in data[0]:
            name = Path(image).name
            self.data_frame.loc[self.data_frame["filename"] == name, "type"] = "train"

        for image in data[2]:
            name = Path(image).name
            self.data_frame.loc[self.data_frame["filename"] == name, "type"] = "val"

        if save:
            copy_images_and_labels(
                data[0],
                data[1],
                self.root_path / "images" / "train",
                self.root_path / "labels" / "train",
            )
            copy_images_and_labels(
                data[2],
                data[3],
                self.root_path / "images" / "val",
                self.root_path / "labels" / "val",
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
        self.data_frame.sort_values(by="type", inplace=True, ignore_index=True)
        return data


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

    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

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
    if type(imgdir) == "str":
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
    split=0.7,
    shuffle=True,
    recurse=True,
):
    label_paths = collect_files_with_suffix(
        f".{label_format.lower()}", labels_dir, recurse=recurse
    )
    image_paths = []
    for suffix in image_format:
        image_paths.extend(
            collect_files_with_suffix(f".{suffix.lower()}", images_dir, recurse=recurse)
        )

    if shuffle:
        random.shuffle(label_paths)

    label_path_map = dict()
    for path in label_paths:
        label_path_map[path.stem] = str(path)

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
            raise ValueError(f"Label for '{label_path.name}' not found")

    split = int(len(all_paths) * split)
    train_images = []
    train_labels = []
    for data in all_paths[:split]:
        train_images.append(data["image"])
        train_labels.append(data["label"])

    val_images = []
    val_labels = []
    for data in all_paths[split:]:
        val_images.append(data["image"])
        val_labels.append(data["label"])

    for image, label in zip(train_images, train_labels):
        if image in val_images or label in val_labels:
            val_images.remove(image)
            val_labels.remove(label)

    return train_images, train_labels, val_images, val_labels


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

def split_data_by_image():
    pass

def split_data_by_collection():
    pass

def split_data_k_fold(k):
    # return generator
    pass
