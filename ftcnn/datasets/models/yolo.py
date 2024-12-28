import random
from collections.abc import Callable
from os import PathLike
from pathlib import Path

from geopandas import gpd
from tqdm.auto import trange

from ftcnn.datasets.tools import make_ndvi_difference_dataset
from ftcnn.geospacial import DataFrameLike
from ftcnn.geospacial.utils import encode_classes, encode_default_classes
from ftcnn.io import save_as_csv, save_as_gpkg, save_as_shp
from ftcnn.modeling.yolo.conversion import geodataframe_to_yolo
from ftcnn.utils import NUM_CPU


def create_ndvi_difference_dataset(
    self,
    source: PathLike | DataFrameLike,
    images_dir: PathLike,
    output_dir: PathLike,
    *,
    background: None | PathLike | DataFrameLike | bool = None,
    background_ratio: float = 1.0,
    split: str | bool = "all",  # 'all', 'collection', True/False
    split_ratio: float = 0.7,  # 0.7 (70/30)
    shuffle_split: bool = True,  # True/False
    shuffle_background: bool = True,  # True/False
    generate_train_data: bool = True,  # True/False
    imgsz: None | int | tuple[int, ...] = 640,
    translate_xy: bool = True,  # True/False
    class_encoder: None | Callable = None,  # None or callback(row)
    exist_ok: bool = False,  # True/False
    clear_output_dir: bool = True,  # True/False
    save_shp: bool = True,  # True/False
    save_gpkg: bool = True,  # True/False
    save_csv: bool = True,  # True/False
    pbar_leave: bool = False,  # True/False
    convert_to_png: bool = True,
):
    pass


def make_yolo_ndvi_difference_dataset(
    shp_file,
    ndvi_dir,
    output_dir,
    *,
    years=None,
    start_year_col="start_year",
    end_year_col="end_year",
    geom_col="geometry",
    tile_size=None,
    clean_dest=False,
    xy_to_index=True,
    encoder=encode_default_classes,
    exist_ok=False,
    save_csv=False,
    save_shp=False,
    save_gpkg=False,
    generate_labels=True,
    tif_to_png=True,
    use_segments=True,
    generate_train_data=True,
    split=0.75,
    split_mode="all",
    shuffle_split=True,
    shuffle_background=True,
    background_bias=None,
    pbar_leave=True,
    num_workers=None,
):
    """
    Converts NDVI (Normalized Difference Vegetation Index) data into a YOLO-compatible dataset format.

    Parameters:
        shp_file (str): Path to the shapefile containing the polygons.
        ndvi_dir (str): Directory containing the NDVI image files.
        output_dir (str): Directory where the output dataset will be saved.
        years (list of int, optional): A list of years to process. Defaults to None.
        start_year_col (str, optional): Column name for the start year. Defaults to "start_year".
        end_year_col (str, optional): Column name for the end year. Defaults to "end_year".
        geom_col (str, optional): Column name for the geometry data. Defaults to "geometry".
        tile_size (int, optional): Size of the tiles for the NDVI data. Defaults to None.
        clean_dest (bool, optional): Whether to clean the destination directory before saving. Defaults to False.
        xy_to_index (bool, optional): Whether to convert coordinates to an index. Defaults to True.
        encoder (function, optional): Function for encoding class labels. Defaults to `encode_default_classes`.
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to False.
        save_csv (bool, optional): Whether to save the dataset as a CSV file. Defaults to False.
        save_shp (bool, optional): Whether to save the dataset as a shapefile. Defaults to False.
        save_gpkg (bool, optional): Whether to save the dataset as a geopackage. Defaults to False.
        generate_labels (bool, optional): Whether to generate labels for the dataset. Defaults to True.
        tif_to_png (bool, optional): Whether to convert TIFF images to PNG format. Defaults to True.
        use_segments (bool, optional): Whether to use segments in the dataset. Defaults to True.
        generate_train_data (bool, optional): Whether to generate training data. Defaults to True.
        split (float, optional): Proportion of data to use for training. Defaults to 0.75.
        split_mode (str, optional): Mode of data splitting ("all", "random", etc.). Defaults to "all".
        shuffle_split (bool, optional): Whether to shuffle the data when splitting. Defaults to True.
        shuffle_background (bool, optional): Whether to shuffle background images. Defaults to True.
        background_bias (float, optional): Bias factor for background data. Defaults to None.
        min_labels_required (int, optional): Minimum number of labels required for a valid dataset. Defaults to 10.
        pbar_leave (bool, optional): Whether to leave the progress bar after completion. Defaults to True.
        num_workers (int, optional): Number of worker processes for parallel processing. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - yolo_ds (YOLODataset): The YOLO dataset object.
            - train_data (optional): The training data if generated.

    Example:
        yolo_ds, train_data = ndvi_to_yolo_dataset("path/to/shapefile.shp", "path/to/ndvi_dir", "path/to/output_dir", years=[2020, 2021], generate_labels=True)
    """
    gdf, (meta_dir, tiles_dir, output_fname) = make_ndvi_difference_dataset(
        shp_file,
        ndvi_dir,
        output_dir,
        years=years,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        geom_col=geom_col,
        tile_size=tile_size,
        clean_dest=clean_dest,
        xy_to_index=xy_to_index,
        exist_ok=exist_ok,
        save_csv=save_csv,
        save_shp=save_shp,
        save_gpkg=False,
        tif_to_png=tif_to_png,
        pbar_leave=False,
        num_workers=num_workers,
    )

    csv_dir = meta_dir / "csv"
    shp_dir = meta_dir / "shp"

    n_calls = 3
    n_calls += 1 if generate_labels else 0
    n_calls += 1 if generate_train_data else 0
    pbar = trange(
        n_calls, desc="Creating YOLO dataset - Encoding classes", leave=pbar_leave
    )

    gdf = encode_classes(gdf, encoder)
    labeled_images = gdf.loc[gdf["class_id"] != -1].values.tolist()

    if background_bias is None:
        new_rows = labeled_images
    else:
        background_images = gdf.loc[gdf["class_id"] == -1].values.tolist()
        if shuffle_background:
            random.shuffle(background_images)
        background_images = background_images[
            : int(len(labeled_images) * background_bias)
        ]

        new_rows = labeled_images + background_images
        print(
            "Number of labeled images",
            len(labeled_images),
            "\nNumber of background images",
            len(background_images),
        )

    gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)

    if save_csv or save_shp:
        output_fname = Path(f"{output_fname}_encoded")
        if save_csv:
            save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / output_fname.with_suffix(".shp"),
            )
        if save_gpkg:
            save_as_gpkg(
                gdf,
                shp_dir / output_fname.with_suffix(".gpkg"),
            )
    pbar.update()

    pbar.set_description(
        f"Creating YOLO dataset - Creating YOLODataset with {len(gdf)} labels"
    )
    yolo_ds = geodataframe_to_yolo(gdf)
    pbar.set_description("Creating YOLO dataset - Dataset created")
    pbar.update()

    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    yolo_ds.generate_yaml_file(
        root_abs_path=output_dir,
        dest_abs_path=output_dir / "config",
        train_path="images/train",
        val_path="images/val",
    )

    train_data = None
    if generate_labels or generate_train_data:
        pbar.update()
        pbar.set_description("Creating YOLO dataset - Generating labels")

        yolo_ds.generate_label_files(
            dest_path=output_dir / "labels" / "generated",
            clear_dir=clean_dest,
            overwrite_existing=exist_ok,
            use_segments=use_segments,
        )
        if generate_train_data:
            pbar.update()
            pbar.set_description(
                "Creating YOLO dataset - Splitting dataset and copying files"
            )

            ds_images_dir = (
                output_dir / "images" / "png-tiles" if tif_to_png else tiles_dir
            )
            train_data = yolo_ds.split_data(
                images_dir=ds_images_dir,
                labels_dir=output_dir / "labels" / "generated",
                split=split,
                shuffle=shuffle_split,
                recurse=True,
                mode=split_mode,
            )

            yolo_df = yolo_ds.data_frame
            yolo_ds.compile(NUM_CPU)
            yolo_ds.data_frame = yolo_df

    if save_csv:
        yolo_ds.to_csv(csv_dir / "yolo_ds.csv")

    pbar.update()
    pbar.set_description("Complete")
    pbar.close()

    return yolo_ds, train_data
