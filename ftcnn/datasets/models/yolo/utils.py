from typing import Any

from tqdm.auto import trange

from ftcnn.datasets.models.tools import build_ndvi_difference_dataset
from ftcnn.modeling.yolo.conversion import geodataframe_to_yolo
from ftcnn.utils import NUM_CPU


def create_ndvi_difference_dataset(cls, config):
    pbar_leave = config["pbar_leave"]
    num_workers = config["num_workers"]
    generate_labels = config["generate_labels"]
    generate_train_data = config["generate_train_data"]
    generate_data = generate_labels or generate_train_data

    config["pbar_leave"] = False
    gdf = build_ndvi_difference_dataset(config)

    total_updates = 1
    total_updates += 1 if generate_data else 0
    total_updates += 1 if generate_train_data else 0
    pbar = trange(
        total_updates,
        desc=f"Creating YOLO dataset - Creating YOLODataset with {len(gdf.keys())} labels",
        leave=pbar_leave,
    )
    yolo_ds = geodataframe_to_yolo(gdf)
    ndvi_ds = cls(
        labels=yolo_ds.labels,
        images=yolo_ds.images,
        compile=True,
        num_workers=num_workers,
    )
    ndvi_ds.config = config
    ndvi_ds.generate_yaml_file(
        root_abs_path=config["root_dir"],
        dest_abs_path=config["config_dir"],
        train_path="images/train",
        val_path="images/val",
    )

    if generate_data:
        pbar.update()
        pbar.set_description(
            "Creating YOLO NDVI difference dataset - Generating labels"
        )

        ndvi_ds.generate_label_files(
            dest_path=config["labels_dir"] / "generated",
            clear_dir=config["clear_output_dir"],
            overwrite_existing=config["exist_ok"],
            use_segments=config["use_segments"],
        )
        if generate_train_data:
            pbar.update()
            pbar.set_description(
                "Creating YOLO NDVI difference dataset - Splitting dataset and copying files"
            )

            ds_images_dir = (
                config["images_dir"] / "png-tiles"
                if config["convert_to_png"]
                else config["images_dir"] / "tiles"
            )
            ndvi_ds.train_data = ndvi_ds.split_data(
                images_dir=ds_images_dir,
                labels_dir=config["labels_dir"] / "generated",
                split=config["split_ratio"],
                shuffle=config["shuffle_split"],
                recurse=True,
                mode=config["split"],
            )

            tmp_df = ndvi_ds.data_frame
            ndvi_ds.compile(NUM_CPU)
            ndvi_ds.data_frame = tmp_df

    if config["save_csv"]:
        ndvi_ds.to_csv(config["meta_dir"] / "csv" / "yolo_ds.csv")

    pbar.set_description("Created YOLO NDVI difference dataset")
    pbar.update()
    pbar.close()
    return ndvi_ds
