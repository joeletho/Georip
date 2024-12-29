from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

from ftcnn.datasets.models.yolo.utils import create_ndvi_difference_dataset
from ftcnn.geospacial import DataFrameLike
from ftcnn.geospacial.utils import encode_default_classes
from ftcnn.modeling.utils import AnnotatedLabel, DatasetSplitMode, ImageData
from ftcnn.modeling.yolo import YOLODatasetBase

__all__ = ["YOLONDVIDifferenceDataset"]


class YOLONDVIDifferenceDataset(YOLODatasetBase):
    config: None | dict[str, Any] = None
    train_data: (
        None
        | tuple[
            tuple[list[ImageData], list[AnnotatedLabel]],
            tuple[list[ImageData], list[AnnotatedLabel]],
        ]
    ) = None

    def __init__(
        self,
        images: list[ImageData],
        labels: list[AnnotatedLabel],
        *,
        compile: bool = True,
        num_workers: int = 8,
    ):
        super().__init__(
            images=images, labels=labels, compile=compile, num_workers=num_workers
        )

    @staticmethod
    def create(
        source: PathLike | DataFrameLike,
        images_dir: PathLike,
        output_dir: PathLike,
        *,
        year_start_column: str = "start_year",
        year_end_column: str = "end_year",
        geometry_column: str = "geometry",
        years: None | tuple[int, int] = None,
        background: None | PathLike | DataFrameLike | bool = None,
        background_ratio: float = 1.0,
        split: DatasetSplitMode = DatasetSplitMode.All,
        split_ratio: float = 0.7,  # 0.7 (70/30)
        shuffle_split: bool = True,  # True/False
        shuffle_background: bool = True,  # True/False
        generate_labels: bool = True,
        generate_train_data: bool = True,  # True/False
        tile_size: None | int | tuple[int, int] = 640,
        translate_xy: bool = True,  # True/False
        class_encoder: Callable = encode_default_classes,  # None or callback(row)
        exist_ok: bool = False,  # True/False
        clear_output_dir: bool = True,  # True/False
        save_shp: bool = True,  # True/False
        save_gpkg: bool = True,  # True/False
        save_csv: bool = True,  # True/False
        pbar_leave: bool = False,  # True/False
        convert_to_png: bool = True,
        use_segments: bool = True,
        num_workers: int = 8,
    ):
        root_dir = Path(output_dir)
        images_dir = Path(images_dir)
        labels_dir = images_dir.parent / "labels"
        config = {
            "source": source,
            "root_dir": root_dir,
            "config_dir": root_dir / "config",
            "images_dir": images_dir,
            "labels_dir": labels_dir / "labels",
            "meta_dir": root_dir / "meta",
            "year_start_column": year_start_column,
            "year_end_column": year_end_column,
            "geometry_column": geometry_column,
            "years": years,
            "background": background,
            "background_ratio": background_ratio,
            "split": split,
            "split_ratio": split_ratio,
            "shuffle_split": shuffle_split,
            "shuffle_background": shuffle_background,
            "generate_labels": generate_labels,
            "generate_train_data": generate_train_data,
            "tile_size": tile_size,
            "translate_xy": translate_xy,
            "class_encoder": class_encoder,
            "exist_ok": exist_ok,
            "clear_output_dir": clear_output_dir,
            "save_shp": save_shp,
            "save_gpkg": save_gpkg,
            "save_csv": save_csv,
            "pbar_leave": pbar_leave,
            "convert_to_png": convert_to_png,
            "use_segments": use_segments,
            "num_workers": num_workers,
        }
        return create_ndvi_difference_dataset(YOLONDVIDifferenceDataset, config)
