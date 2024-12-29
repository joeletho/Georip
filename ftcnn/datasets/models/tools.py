import random
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from tqdm.auto import trange

from ftcnn.datasets.tools import make_ndvi_difference_dataset
from ftcnn.geospacial.utils import encode_classes
from ftcnn.io import save_as_csv, save_as_gpkg, save_as_shp
from ftcnn.utils import FTCNN_TMP_DIR


def build_ndvi_difference_dataset(config: dict[str, Any]):
    source = config["source"]
    images_dir = config["images_dir"]
    root_dir = config["root_dir"]
    year_start_column = config["year_start_column"]
    year_end_column = config["year_end_column"]
    geometry_column = config["geometry_column"]
    years = config["years"]
    background = config["background"]
    background_ratio = config["background_ratio"]
    shuffle_background = config["shuffle_background"]
    tile_size = config["tile_size"]
    translate_xy = config["translate_xy"]
    class_encoder = config["class_encoder"]
    exist_ok = config["exist_ok"]
    clear_output_dir = config["clear_output_dir"]
    save_shp = config["save_shp"]
    save_gpkg = config["save_gpkg"]
    save_csv = config["save_csv"]
    pbar_leave = config["pbar_leave"]
    convert_to_png = config["convert_to_png"]
    num_workers = config["num_workers"]

    if isinstance(source, pd.DataFrame) or isinstance(source, gpd.GeoDataFrame):
        if isinstance(source, pd.DataFrame):
            source = gpd.GeoDataFrame(source)
        tmp_path = FTCNN_TMP_DIR / f"ndvi_difference_dataset_{datetime.now()}.shp"
        save_as_shp(source, tmp_path)
        source = tmp_path

    total_updates = 4
    pbar = trange(
        total_updates,
        desc="Creating NDVI Difference dataset",
        leave=pbar_leave,
    )

    # TODO: process background images/geometry here. Need to handle:
    #   1. `background` is None or False, indicating only return images with valid geometry (no background)
    #   2. `background` is a shapefile, which means:
    #       a) `background` may contain geometry, or
    #       b) `background` can be any row with an `geotiff_filename_column` column as background
    #   3. `background` is a dataframe, so we can query the filename in `geotiff_filename_column` column

    ds_name, gdf = make_ndvi_difference_dataset(
        source,
        images_dir,
        root_dir,
        years=years,
        start_year_col=year_start_column,
        end_year_col=year_end_column,
        geom_col=geometry_column,
        tile_size=tile_size,
        clean_dest=clear_output_dir,
        translate_xy=translate_xy,
        exist_ok=exist_ok,
        save_csv=save_csv,
        save_shp=save_shp,
        save_gpkg=save_gpkg,
        convert_to_png=convert_to_png,
        pbar_leave=pbar_leave,
        num_workers=num_workers,
    )

    pbar.set_description("Creating NDVI Difference dataset - Encoding classes")
    pbar.update()

    gdf = encode_classes(gdf, class_encoder)

    pbar.set_description("Creating NDVI Difference dataset - Filtering background")
    pbar.update()

    labeled_images = gdf.loc[gdf["class_id"] != -1].values.tolist()

    if background is None:
        new_rows = labeled_images
    else:
        background_images = gdf.loc[gdf["class_id"] == -1].values.tolist()
        if shuffle_background:
            random.shuffle(background_images)
        background_images = background_images[
            : int(len(labeled_images) * background_ratio)
        ]

        new_rows = labeled_images + background_images
        print(
            "Number of labeled images",
            len(labeled_images),
            "\nNumber of background images",
            len(background_images),
        )

    pbar.set_description("Creating NDVI Difference dataset - Finishing up")
    pbar.update()

    gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)

    if save_csv or save_shp:
        meta_dir = config["meta_dir"]
        ds_name = Path(f"{ds_name}_encoded")
        if save_csv:
            save_as_csv(gdf, meta_dir / "csv" / ds_name.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                meta_dir / "shp" / ds_name.with_suffix(".shp"),
            )
        if save_gpkg:
            save_as_gpkg(
                gdf,
                meta_dir / "shp" / ds_name.with_suffix(".gpkg"),
            )
    pbar.update()
    pbar.close()

    return gdf
