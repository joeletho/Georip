import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from tqdm.auto import trange

from ftcnn.datasets.tools import make_ndvi_difference_dataset
from ftcnn.datasets.utils import (
    TMP_FILE_PREFIX,
    gdf_ndvi_validate_years_as_ints,
    postprocess_geo_source,
    preprocess_geo_source,
)
from ftcnn.geospacial.utils import encode_classes
from ftcnn.io import save_as_csv, save_as_gpkg, save_as_shp
from ftcnn.utils import FTCNN_TMP_DIR


def build_ndvi_difference_dataset(config: dict[str, Any]):
    source = config["source"]
    source_images_dir = config["source_images_dir"]
    root_dir = config["root_dir"]
    region_column = config["region_column"]
    year_start_column = config["year_start_column"]
    year_end_column = config["year_end_column"]
    geometry_column = config["geometry_column"]
    years = config["years"]
    background = config["background"]
    background_ratio = config["background_ratio"]
    tile_size = config["tile_size"]
    translate_xy = config["translate_xy"]
    class_encoder = config["class_encoder"]
    background_filter = config["background_filter"]
    exist_ok = config["exist_ok"]
    clear_output_dir = config["clear_output_dir"]
    save_shp = config["save_shp"]
    save_gpkg = config["save_gpkg"]
    save_csv = config["save_csv"]
    pbar_leave = config["pbar_leave"]
    convert_to_png = config["convert_to_png"]
    num_workers = config["num_workers"]
    preserve_fields = config["preserve_fields"]

    total_updates = 3
    pbar = trange(
        total_updates,
        desc="Creating NDVI Difference dataset",
        leave=pbar_leave,
    )
    timestamp = f"{time.time()}"
    timestamp = timestamp[: timestamp.find(".")]

    source_path = source
    if isinstance(source, pd.DataFrame) or isinstance(source, gpd.GeoDataFrame):
        gdf_ndvi_validate_years_as_ints(
            source,
            start_year_column=year_start_column,
            end_year_column=year_end_column,
        )

        if isinstance(source, pd.DataFrame):
            source = gpd.GeoDataFrame(source)
        source_path = (
            FTCNN_TMP_DIR / f"{TMP_FILE_PREFIX}ndvi_difference_dataset_{timestamp}.shp"
        )
        save_as_shp(source, source_path)

    source_path = preprocess_geo_source(source_path, geometry_column)
    if not isinstance(region_column, list):
        region_column = [region_column]

    if preserve_fields is None:
        preserve_fields = [*region_column]
    else:
        preserve_fields.extend(region_column)

    ds_name, gdf = make_ndvi_difference_dataset(
        source_path,
        source_images_dir,
        root_dir,
        years=years,
        region_col=region_column,
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
        preserve_fields=preserve_fields,
    )

    postprocess_geo_source(source_path)

    pbar.update()

    gdf = encode_classes(gdf, class_encoder)

    if background:
        if not callable(background_filter):
            raise ValueError(
                "`background_filter` must be a callable function when `background` is True."
            )

        background_gdf = gdf.iloc[gdf.apply(background_filter, axis=1)]
        truth_gdf = gdf.drop(index=background_gdf.index)

        sample_size = int(len(truth_gdf) * background_ratio)
        background_gdf = background_gdf.sample(n=sample_size)
        print(
            f"Number of labeled images: {len(truth_gdf)}\n"
            f"Number of background images: {len(background_gdf)}"
        )
        gdf = gpd.GeoDataFrame(pd.concat([truth_gdf, background_gdf]), crs=gdf.crs)

    pbar.set_description("Creating NDVI Difference dataset - Finishing up")
    pbar.update()

    if save_csv or save_shp:
        meta_dir = config["meta_dir"]
        dir_name = source_path.stem.replace(TMP_FILE_PREFIX, "")
        ds_name = Path(f"{ds_name}_encoded")
        if save_csv:
            save_as_csv(gdf, meta_dir / "csv" / dir_name / ds_name.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                meta_dir / "shp" / dir_name / ds_name.with_suffix(".shp"),
            )
        if save_gpkg:
            save_as_gpkg(
                gdf,
                meta_dir / "shp" / dir_name / ds_name.with_suffix(".gpkg"),
            )
    pbar.update()
    pbar.close()

    return gdf
