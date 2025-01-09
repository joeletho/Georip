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
        source_images_dir: PathLike,
        output_dir: PathLike,
        *,
        region_column: str,
        year_start_column: str,
        year_end_column: str,
        geometry_column: str = "geometry",
        years: None | tuple[int, int] = None,
        background: bool = False,
        background_ratio: float = 1.0,
        background_filter: Callable | None = None,
        split: DatasetSplitMode = DatasetSplitMode.All,
        split_ratio: float = 0.7,  # 0.7 (70/30)
        shuffle_split: bool = True,  # True/False
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
        preserve_fields: list[str | dict[str, str]] | None = None,
    ):
        root_dir = Path(output_dir)
        source_images_dir = Path(source_images_dir)
        images_dir = root_dir / "images"
        labels_dir = root_dir / "labels"
        config = {
            "source": source,
            "root_dir": root_dir,
            "config_dir": root_dir / "config",
            "source_images_dir": source_images_dir,
            "images_dir": images_dir,
            "labels_dir": labels_dir,
            "meta_dir": root_dir / "meta",
            "region_column": region_column,
            "year_start_column": year_start_column,
            "year_end_column": year_end_column,
            "geometry_column": geometry_column,
            "years": years,
            "background": background,
            "background_ratio": background_ratio,
            "background_filter": background_filter,
            "split": split,
            "split_ratio": split_ratio,
            "shuffle_split": shuffle_split,
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
            "preserve_fields": preserve_fields,
        }
        return create_ndvi_difference_dataset(YOLONDVIDifferenceDataset, config)
