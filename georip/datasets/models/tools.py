import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from tqdm.auto import trange

from georip.datasets.tools import make_ndvi_difference_dataset
from georip.datasets.utils import (
    TMP_FILE_PREFIX,
    encode_classes,
    merge_source_and_background,
    postprocess_geo_source,
    preprocess_geo_source,
)
from georip.geoprocessing.utils import gdf_ndvi_validate_years_as_ints
from georip.io import save_as_csv, save_as_gpkg, save_as_shp
from georip.utils import _WRITE_LOCK, GEORIP_TMP_DIR


def build_ndvi_difference_dataset(config: dict[str, Any]):
    shapefile = config["shapefile"]
    image_dir = config["image_dir_src"]
    output_dir = config["output_dir"]
    region_column = config["region_column"]
    year_start_column = config["year_start_column"]
    year_end_column = config["year_end_column"]
    geometry_column = config["geometry_column"]
    years = config["years"]
    background_ratio = config["background_ratio"]
    background_filter = config["background_filter"]
    background_seed = config["background_seed"]
    tile_size = config["tile_size"]
    stride = config["stride"]
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
    preserve_fields = config["preserve_fields"]

    if background_filter is not None and background_ratio <= 0:
        raise ValueError("Background ratio must be greater than 0")

    total_updates = 4
    total_updates += 1 if config["background_shapefile"] else 0
    pbar = trange(
        total_updates,
        leave=pbar_leave,
    )
    timestamp = f"{time.time()}"
    timestamp = timestamp[: timestamp.find(".")]

    if config["background_shapefile"]:
        pbar.set_description("Merging source and background data")
        source_path = merge_source_and_background(config)
        pbar.update()
    else:
        source_path = shapefile

    pbar.set_description("Preprocessing shapefile")

    if isinstance(shapefile, pd.DataFrame) or isinstance(shapefile, gpd.GeoDataFrame):
        gdf_ndvi_validate_years_as_ints(
            shapefile,
            start_year_column=year_start_column,
            end_year_column=year_end_column,
        )

        if isinstance(shapefile, pd.DataFrame):
            shapefile = gpd.GeoDataFrame(shapefile)
        source_path = (
            GEORIP_TMP_DIR / f"{TMP_FILE_PREFIX}ndvi_difference_dataset_{timestamp}.shp"
        )
        save_as_shp(shapefile, source_path)

    source_path = preprocess_geo_source(source_path, geometry_column)

    if not isinstance(region_column, list):
        region_column = [region_column]

    if preserve_fields is None:
        preserve_fields = [*region_column]
    else:
        if not isinstance(preserve_fields, list):
            preserve_fields = [preserve_fields]
        preserve_fields.extend(region_column)

    pbar.update()
    pbar.set_description("Creating NDVI Difference dataset")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            make_ndvi_difference_dataset,
            source_path,
            image_dir,
            output_dir,
            years=years,
            region_col=region_column,
            start_year_col=year_start_column,
            end_year_col=year_end_column,
            geom_col=geometry_column,
            tile_size=tile_size,
            stride=stride,
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

        while True:
            if future.done():
                ds_name, gdf = future.result()
                break
            time.sleep(1)
            pbar.refresh()

    postprocess_geo_source(source_path)

    pbar.update()

    if class_encoder is None:
        encoder_params = {
            "geom_col": geometry_column,
            "class_col": config["class_column"],
            "class_names": config["class_names"],
        }
        gdf = encode_classes(gdf, None, **encoder_params)
    else:
        gdf = encode_classes(gdf, class_encoder)

    if isinstance(background_filter, bool) or background_filter is not None:
        if background_filter is True:

            def _filter_background(row):
                return row["class_name"] not in config["class_names"]

            background_filter = _filter_background
        elif not callable(background_filter):
            raise ValueError(
                f"`background_filter` must be callable, not {type(background_filter)}"
            )

        background_gdf = gdf.loc[gdf.apply(background_filter, axis=1)]
        if background_gdf.empty:
            raise ValueError("No background found after filtering")

        truth_gdf = gdf.drop(index=background_gdf.index)

        n_sample = int(len(truth_gdf) * background_ratio)
        if background_gdf is not None:
            # We need to be sure we don't exceed the number of background rows
            n_sample = min(len(background_gdf), n_sample)
        background_gdf = background_gdf.sample(
            n=n_sample, random_state=background_seed, ignore_index=True
        )

        lock_id = _WRITE_LOCK.acquire()
        print(
            f"Number of labeled images: {len(truth_gdf)}\n"
            f"Number of background images: {len(background_gdf)}"
        )
        _WRITE_LOCK.free(lock_id)
        gdf = gpd.GeoDataFrame(pd.concat([truth_gdf, background_gdf]), crs=gdf.crs)

    gdf.set_geometry(geometry_column)

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
