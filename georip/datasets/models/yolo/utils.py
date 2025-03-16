from concurrent.futures import ThreadPoolExecutor
from time import sleep

from georip.datasets.models.tools import build_ndvi_difference_dataset
from georip.io import clear_directory, save_as_yaml
from georip.modeling.yolo.conversion import geodataframe_to_yolo
from georip.utils import NUM_CPU
from tqdm.auto import trange


def create_ndvi_difference_dataset(cls, config):
    # Clear now so we can start to save data
    if config["clear_output_dir"] and config["output_dir"].exists():
        clear_directory(config["output_dir"])

    yaml_path = config["config_dir"] / "args.yaml"
    save_as_yaml(
        config,
        yaml_path,
        parents=True,
        exist_ok=True,
        message=f"Argument list saved to {yaml_path}",
    )

    config["clear_output_dir"] = False

    pbar_leave = config["pbar_leave"]
    num_workers = config["num_workers"]
    generate_labels = config["generate_labels"]
    generate_train_data = config["generate_train_data"]
    generate_data = generate_labels or generate_train_data

    gdf = build_ndvi_difference_dataset(config)

    total_updates = 1
    total_updates += 1 if generate_data else 0
    pbar = trange(
        total_updates,
        desc=f"Creating YOLO dataset - Creating YOLODataset with {len(gdf)} labels",
        leave=pbar_leave,
    )
    yolo_ds = geodataframe_to_yolo(gdf, config["geometry_column"])

    if config["save_csv"]:
        yolo_ds.to_csv(config["meta_dir"] / "csv" / "yolo_ds_base.csv")

    ndvi_ds = cls(
        labels=yolo_ds.labels,
        images=yolo_ds.images,
        compile=True,
        num_workers=num_workers,
    )

    if config["save_csv"]:
        ndvi_ds.to_csv(config["meta_dir"] / "csv" / "yolo_ds_ndvi.csv")

    ndvi_ds.config = config
    ndvi_ds.generate_yaml_file(
        root_abs_path=config["output_dir"],
        dest_abs_path=config["config_dir"],
    )

    if generate_data:
        pbar.update()
        pbar.set_description(
            "Creating YOLO NDVI difference dataset - Generating labels"
        )

        with ThreadPoolExecutor(max_workers=1) as executor:

            def process_generate_labels():
                ndvi_ds.generate_label_files(
                    dest_path=config["label_dir_dest"] / "generated",
                    clear_dir=False,
                    overwrite_existing=config["exist_ok"],
                    use_segments=config["use_segments"],
                )

            future = executor.submit(process_generate_labels)
            while True:
                if future.done():
                    break
                sleep(1)
                pbar.refresh()

            pbar.update()
            pbar.set_description(
                "Creating YOLO NDVI difference dataset - Splitting dataset and copying files"
            )

            def process_generate_data():
                ds_images_dir = (
                    config["image_dir_dest"] / "tiles/png"
                    if config["convert_to_png"]
                    else config["image_dir_dest"] / "tiles/tif"
                )
                ndvi_ds.train_data = ndvi_ds.split_data(
                    images_dir=ds_images_dir,
                    labels_dir=config["label_dir_dest"] / "generated",
                    train_split=config["train_split_ratio"],
                    test_split=config["test_split_ratio"],
                    stratify=config["stratify"],
                    shuffle=config["shuffle_split"],
                    shuffle_seed=config["shuffle_seed"],
                    save=generate_train_data,
                    recurse=True,
                    mode=config["split_mode"],
                )

                tmp_df = ndvi_ds.data_frame
                ndvi_ds.compile(NUM_CPU)
                ndvi_ds.data_frame = tmp_df

                return ndvi_ds

            future = executor.submit(process_generate_data)

            while True:
                if future.done():
                    ndvi_ds = future.result()
                    break
                sleep(1)
                pbar.refresh()

    if config["save_csv"]:
        ndvi_ds.to_csv(config["meta_dir"] / "csv" / "yolo_ds_ndvi_final.csv")

    pbar.set_description("Created YOLO NDVI difference dataset")
    pbar.update()
    pbar.close()

    return ndvi_ds
