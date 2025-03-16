from collections.abc import Callable
from pathlib import Path
from typing import Any

from georip.datasets.models.yolo.utils import create_ndvi_difference_dataset
from georip.geoprocessing import DataFrameLike
from georip.modeling.utils import AnnotatedLabel, DatasetSplitMode, ImageData
from georip.modeling.yolo import YOLODatasetBase
from georip.utils import StrPathLike

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
        source: StrPathLike | DataFrameLike,
        image_dir: StrPathLike,
        output_dir: StrPathLike,
        *,
        region_column: str | list[str],
        year_start_column: str,
        year_end_column: str,
        class_column: str,
        class_names: str | list[str],
        geometry_column: str = "geometry",
        years: None | tuple[int, int] = None,
        background_shapefile: StrPathLike | None = None,
        background_ratio: float = 1.0,
        background_filter: bool | Callable | None = None,
        background_seed: int | None = None,
        split_mode: DatasetSplitMode = DatasetSplitMode.All,
        train_split_ratio: float = 0.7,  # 0.7 (70/30)
        test_split_ratio: float = 0.15,
        shuffle_split: bool = True,  # True/False
        shuffle_seed: int | None = None,
        stratify: bool = True,
        generate_labels: bool = True,
        generate_train_data: bool = True,  # True/False
        tile_size: None | int | tuple[int, int] = 640,
        stride: None | int | tuple[int, int] = None,
        translate_xy: bool = True,  # True/False
        class_encoder: Callable | None = None,  # None or callback(row)
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
        output_dir = Path(output_dir).resolve()
        image_dir_src = Path(image_dir).resolve()
        image_dir_dest = output_dir / "images"
        label_dir_dest = output_dir / "labels"
        config = {
            "shapefile": source,
            "output_dir": output_dir,
            "config_dir": output_dir / "config",
            "image_dir_src": image_dir_src,
            "image_dir_dest": image_dir_dest,
            "label_dir_dest": label_dir_dest,
            "meta_dir": output_dir / "meta",
            "region_column": region_column,
            "year_start_column": year_start_column,
            "year_end_column": year_end_column,
            "class_column": class_column,
            "class_names": class_names,
            "geometry_column": geometry_column,
            "years": years,
            "background_shapefile": background_shapefile,
            "background_ratio": background_ratio,
            "background_filter": background_filter,
            "background_seed": background_seed,
            "split_mode": split_mode,
            "train_split_ratio": train_split_ratio,
            "test_split_ratio": test_split_ratio,
            "shuffle_split": shuffle_split,
            "shuffle_seed": shuffle_seed,
            "stratify": stratify,
            "generate_labels": generate_labels,
            "generate_train_data": generate_train_data,
            "tile_size": tile_size,
            "stride": stride,
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
