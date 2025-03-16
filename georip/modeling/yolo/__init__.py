import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import yaml
from geopandas.geodataframe import json
from osgeo.gdal import sys
from PIL import Image, ImageDraw
from torchvision import torch, tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from tqdm.auto import trange

from georip.modeling.yolo import predict
from georip.geometry import normalize_point
from georip.io import clear_directory
from georip.modeling import maskrcnn
from georip.modeling.maskrcnn import collate_fn
from georip.modeling.utils import (AnnotatedLabel, BBox, DatasetSplitMode,
                                  ImageData, Serializable, XYPair,
                                  convert_segment_to_bbox,
                                  copy_images_and_labels,
                                  display_image_and_annotations, make_dataset,
                                  maskrcnn_get_transform,
                                  parse_labels_from_csv)
from georip.utils import NUM_CPU, TQDM_INTERVAL, Lock, StrPathLike

__all__ = ["predict"]


class YOLODatasetBase(Serializable):
    _lock = Lock()
    data_frame: pd.DataFrame
    class_map: dict[str, int]
    class_distribution: dict[str, int]
    root_path: Path = Path()
    is_compiled = False

    def __init__(
        self,
        labels: list[AnnotatedLabel],
        images: list[ImageData],
        *,
        compile: bool = True,
        num_workers: None | int = None,
    ):
        self.labels = labels
        self.images = images

        if compile:
            if num_workers is None or not isinstance(num_workers, int):
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

    def get_num_images_and_labels(self, dataset_type: str):

        return (
            self.data_frame.loc[self.data_frame["type"] == dataset_type, "filename"]
            .unique()
            .shape[0]
        ), self.data_frame.loc[self.data_frame["type"] == dataset_type].shape[0]

    def info(self):
        if not self.is_compiled:
            raise RuntimeError("Must compile dataset before calling method")

        num_train_images, num_train_labels = self.get_num_images_and_labels("train")
        num_val_images, num_val_labels = self.get_num_images_and_labels("val")
        num_test_images, num_test_labels = self.get_num_images_and_labels("test")

        lock_id = self._lock.acquire()
        print(
            "YOLO Dataset information\n"
            + f"Number of labels: {len(self.labels)}\n"
            + f"Number of images: {len(self.images)}\n"
            + f"Number of classes: {self.get_num_classes()}\n"
            + f"Training data: {num_train_images} images, {num_train_labels} labels\n"
            + f"Validation data: {num_val_images} images, {num_val_labels} labels\n"
            + f"Test data: {num_test_images} images, {num_test_labels} labels\n"
        )
        self._lock.free(lock_id)

    def summary(self):
        if not self.is_compiled:
            raise RuntimeError("Must compile dataset before calling method")

        def get_formatted_class_distribution_by_type(dataset_type: str) -> str:
            subset = self.data_frame.loc[self.data_frame["type"] == dataset_type]
            distribution = (
                subset[["class_id", "class_name"]].value_counts().sort_index().to_dict()
            )
            return "\n".join(
                f"    Class {class_id}, {class_name}: {count}"
                for (class_id, class_name), count in distribution.items()
            )

        train_distribution = get_formatted_class_distribution_by_type("train")
        val_distribution = get_formatted_class_distribution_by_type("val")
        test_distribution = get_formatted_class_distribution_by_type("test")

        num_train_images, num_train_labels = self.get_num_images_and_labels("train")
        num_val_images, num_val_labels = self.get_num_images_and_labels("val")
        num_test_images, num_test_labels = self.get_num_images_and_labels("test")

        lock_id = self._lock.acquire()
        print(
            "YOLO Dataset summary\n"
            + f"Number of labels: {len(self.labels)}\n"
            + f"Number of images: {len(self.images)}\n"
            + f"Number of classes: {self.get_num_classes()}\n"
            + "Class distribution:\n"
            + self.get_class_distribution()
            + "\n"
            + f"Training data: {num_train_images} images, {num_train_labels} labels\n"
            + train_distribution
            + "\n"
            + f"Validation data: {num_val_images} images, {num_val_labels} labels\n"
            + val_distribution
            + "\n"
            + f"Test data: {num_test_images} images, {num_test_labels} labels\n"
            + test_distribution
            + "\n"
            + "\n\n"
            + f"Data:\n{self.data_frame}\n"
        )
        self._lock.free(lock_id)

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
            x, y, w, h = normalize_point(
                bbox.x,
                bbox.y,
                imgsize[0],
                imgsize[1],
                xoffset=bbox.width,
                yoffset=bbox.height,
                include_dims=True,
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
        if self.is_compiled:
            return

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
            "dirpath": [],
            "segments": [],
        }
        self.labels = list(set(self.labels))
        self.images = list(set(self.images))

        self.class_map = YOLODatasetBase.get_mapped_classes(self.labels)

        if not (len(self.labels) and len(self.images)):
            raise ValueError("Dataset is empty")
        if not len(self.class_map.keys()):
            raise ValueError("Dataset does not contain classes")

        self.class_distribution = {
            name: 0 for name in sorted(self.class_map, key=lambda x: self.class_map[x])
        }

        indices_to_remove = []

        def __exec__(labels):
            total_updates = len(labels)
            updates = 0
            lock_id = self._lock.acquire()
            pbar = trange(
                total_updates,
                desc="Compiling YOLODataset labels and images",
                leave=False,
            )
            pbar.refresh()
            self._lock.free(lock_id)

            start = time()
            for i, label in enumerate(labels):
                lock_id = self._lock.acquire()
                self.class_distribution[label.class_name] += 1
                self._lock.free(lock_id)

                image_data = None
                for image in self.images:
                    if label.image_filename == image.filename:
                        image_data = image
                        break
                if image_data is None:
                    lock_id = self._lock.acquire()
                    print(
                        f"Image '{label.image_filename}' not found in labels -- label flagged for removal",
                        file=sys.stderr,
                    )
                    self._lock.free(lock_id)
                    indices_to_remove.append(i)
                    continue

                if label.bbox is None:
                    label.bbox = BBox(-1, -1, -1, -1)
                else:
                    YOLODatasetBase.convert_bbox_to_yolo(
                        imgsize=image_data.shape, bbox=label.bbox
                    )

                if label.segments is not None and len(label.segments) > 0:
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
                        label.bbox = YOLODatasetBase.convert_bbox_to_yolo(
                            imgsize=image_data.shape,
                            bbox=convert_segment_to_bbox(label.segments),
                        )
                else:
                    label.segments = ""

                lock_id = self._lock.acquire()
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
                data["dirpath"].append(Path(image_data.filepath).parent)
                data["bbox_x"].append(label.bbox.x if label.bbox.x != -1 else "")
                data["bbox_y"].append(label.bbox.y if label.bbox.y != -1 else "")
                data["bbox_w"].append(
                    label.bbox.width if label.bbox.width != -1 else ""
                )
                data["bbox_h"].append(
                    label.bbox.height if label.bbox.height != -1 else ""
                )
                data["segments"].append(
                    " ".join([str(point) for point in label.segments]),
                )
                self._lock.free(lock_id)

                if time() - start >= TQDM_INTERVAL * NUM_CPU:
                    lock_id = self._lock.acquire()
                    pbar.update()
                    updates += 1
                    start = time()
                    self._lock.free(lock_id)
            if updates < total_updates:
                pbar.update(total_updates - updates)
            pbar.close()

        if len(self.labels) < 100:
            __exec__(self.labels)
        else:
            num_workers = max(1, min(num_workers, NUM_CPU))

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
                        raise msg

        self.data_frame = pd.DataFrame.from_dict(data)
        self.data_frame = self.data_frame.drop_duplicates().sort_values(
            by=["type"], ignore_index=True
        )

        if len(indices_to_remove) > 0:
            lock_id = self._lock.acquire()
            print("Cleaning unused labels ...")
            for counter, index in enumerate(indices_to_remove):
                pop_index = index - counter + 1
                if pop_index > len(self.labels):
                    break
                label = self.labels.pop(pop_index)
                print(f"  Removed: {label}")
            self._lock.free(lock_id)

        self.is_compiled = True
        return self

    def to_csv(self, output_path: StrPathLike, **kwargs):
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
    def from_csv(path: StrPathLike, **kwargs):
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
           directory: str
               the path of the directory containing the image
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
        num_workers = kwargs.pop("num_workers", NUM_CPU)

        df = pd.read_csv(path)
        image_map = {
            row["filename"]: str(Path(row["dirpath"]) / row["filename"])
            for _, row in df.iterrows()
        }
        image_names = set()
        images = []

        pbar = trange(1, desc="Parsing labels from CSV", leave=False)
        labels = parse_labels_from_csv(path, **kwargs)

        total_updates = len(labels) + 1
        updates = 0
        start = time()
        pbar.reset(total_updates)
        pbar.update()

        pbar.set_description("Collecting images")
        pbar.refresh()

        for label in labels:
            image_name = label.image_filename
            if image_name not in image_names:
                image_names.add(image_name)
                image = ImageData(image_map[image_name])
                images.append(image)
            if time() - start >= TQDM_INTERVAL * NUM_CPU:
                pbar.update()
                updates += 1
                start = time()
        if updates < total_updates:
            pbar.update(total_updates - updates)
        pbar.close()

        return YOLODatasetBase(
            labels=labels, images=images, compile=compile, num_workers=num_workers
        )

    def generate_label_files(
        self,
        dest_path: StrPathLike,
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
        if not self.is_compiled:
            raise RuntimeError("Must compile dataset before calling method")

        dest_path = Path(dest_path).resolve()
        if not dest_path.exists():
            dest_path.mkdir(parents=True)

        existing_label_files = glob(str(dest_path / "*.txt"))

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
        lock_id = self._lock.acquire()
        pbar = trange(total_updates, desc="Generating labels")
        pbar.refresh()
        self._lock.free(lock_id)
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
                        if 1 < len(points) < 6:
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

            if _type in ["train", "val", "test"]:
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

        lock_id = self._lock.acquire()
        pbar.set_description("Complete")
        pbar.close()
        print(
            f"Successfully generated {annotations} annoations and {backgrounds} backgrounds to {len(files)} files"
        )
        self._lock.free(lock_id)

    def generate_yaml_file(
        self,
        root_abs_path: StrPathLike,
        dest_abs_path: StrPathLike | None = None,
        *,
        filename: str = "data.yaml",
        train_path: StrPathLike | str = "images/train",
        val_path: StrPathLike | str = "images/val",
        test_path: StrPathLike | str = "images/test",
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

        root_abs_path = Path(root_abs_path)
        dest_abs_path = Path(dest_abs_path).resolve()
        dest_abs_path.mkdir(parents=True, exist_ok=True)
        data_yaml_path = dest_abs_path / filename

        with open(data_yaml_path, "w") as f:
            f.write(f"path: {str(root_abs_path)}\n")
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            f.write(f"test: {test_path}\n")
            f.write("names:\n")
            for name, id in self.class_map.items():
                if name.lower() != "background":
                    f.write(f"  {id}: {name}\n")
        self.root_path = root_abs_path
        self.data_yaml_path = dest_abs_path / filename

    def split_data(
        self,
        images_dir: StrPathLike,
        labels_dir: StrPathLike,
        *,
        train_split: float = 0.7,
        test_split: float = 0.15,
        shuffle: bool = True,
        shuffle_seed: int | None = None,
        stratify: bool = True,
        recurse: bool = True,
        mode: DatasetSplitMode | str = DatasetSplitMode.All,
        save: bool = False,
        clear_dest: bool = False,
        **kwargs,
    ) -> tuple[
        tuple[list[ImageData], list[AnnotatedLabel]],  # Train
        tuple[list[ImageData], list[AnnotatedLabel]],  # Validation
        tuple[list[ImageData], list[AnnotatedLabel]],  # Test
    ]:
        image_class_data = {
            str(row["filename"]): int(row["class_id"])
            for _, row in self.data_frame.iterrows()
        }
        train_ds, val_ds, test_ds = make_dataset(
            images_dir,
            labels_dir,
            mode=mode,
            train_split=train_split,
            test_split=test_split,
            stratify_split=stratify,
            class_data=image_class_data,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            recurse=recurse,
            **kwargs,
        )
        if train_split > 0.5:
            if len(train_ds[0]) < len(val_ds[0]):
                train_ds, val_ds = val_ds, train_ds
            if len(train_ds[0]) < len(test_ds[0]):
                train_ds, test_ds = test_ds, train_ds

        def assign_data_class_info(
            ds: tuple[list[ImageData], list[AnnotatedLabel]], dataset_type: str
        ):
            for image in ds[0]:
                name = str(image.filename)
                self.data_frame.loc[self.data_frame["filename"] == name, "type"] = (
                    dataset_type
                )

        assign_data_class_info(train_ds, "train")
        assign_data_class_info(val_ds, "val")
        assign_data_class_info(test_ds, "test")

        def copy_dataset(
            ds: tuple[list[ImageData], list[AnnotatedLabel]],
            dataset_type: str,
            clear_dest: bool = False,
        ):
            images_dest = self.root_path / "images" / dataset_type
            labels_dest = self.root_path / "labels" / dataset_type
            if clear_dest:
                clear_directory(images_dest)
                clear_directory(labels_dest)

            copy_images_and_labels(
                image_paths=[image.filepath for image in ds[0]],
                label_paths=[label.filepath for label in ds[1]],
                images_dest=images_dest,
                labels_dest=labels_dest,
            )

        if save:
            copy_dataset(train_ds, "train", clear_dest=clear_dest)
            copy_dataset(val_ds, "val", clear_dest=clear_dest)
            copy_dataset(test_ds, "test", clear_dest=clear_dest)

            self.data_frame = pd.DataFrame(
                pd.concat(
                    [
                        self.data_frame.loc[self.data_frame["type"] == "train"],
                        self.data_frame.loc[self.data_frame["type"] == "val"],
                        self.data_frame.loc[self.data_frame["type"] == "test"],
                    ],
                    ignore_index=True,
                )
            )
            self.data_frame = self.data_frame.sort_values(
                by=["type", "class_id"], ignore_index=True
            )
            self.images = train_ds[0] + val_ds[0] + test_ds[0]
            self.labels = train_ds[1] + val_ds[1] + test_ds[1]

        return train_ds, val_ds, test_ds


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
