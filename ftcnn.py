import io
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import ArgumentError
from pathlib import Path
from types import FunctionType

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import shapely
import skimage.io as skio
from osgeo import gdal, gdal_array
from PIL import Image
from rasterio.windows import Window
from shapely import normalize, wkt
from shapely.geometry import Polygon
from skimage import img_as_float
from tqdm.auto import tqdm, trange

from .modeling.types import YOLODataset
from .utils import (clear_directory, collect_files_with_suffix, get_cpu_count,
                    linterp, pathify)

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)
gdal.UseExceptions()


def save_as_shp(gdf: gpd.GeoDataFrame, path, exist_ok=False):
    path = pathify(path)
    if not exist_ok and path.exists():
        raise FileExistsError(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(path, driver="ESRI shapefile")


def save_as_csv(df: pd.DataFrame | gpd.GeoDataFrame, path, exist_ok=False):
    path = pathify(path)
    if not exist_ok and path.exists():
        raise FileExistsError(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


def load_shapefile(path) -> gpd.GeoDataFrame:
    if Path(path).suffix == ".csv":
        df = gpd.read_file(path)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        shp_df = gpd.GeoDataFrame(df)
    else:
        shp_df = gpd.read_file(path)
    return shp_df


def open_tif(path, *, masked=True):
    return rxr.open_rasterio(path, masked=masked).squeeze()


def assign_default_classes(row):
    geom = row.get("geometry")
    return (
        ("0", "Treatment")
        if geom is not None and not geom.is_empty and geom.area > 1
        else ("-1", "Background")
    )


def parse_filename(series: pd.Series):
    subregion = str(series["Subregion"])
    startyear = str(series["StartYear"])
    endyear = str(series["EndYear"])

    years_part = "to".join([startyear, endyear])
    end_part = "NDVI_Difference.tif"

    filename = subregion
    last = filename[-1]
    if last.isdigit():
        filename += "_"
    elif last == "E":
        filename = "_".join([filename[:-1], "Expanded", ""])
    start_part = filename + years_part
    return "_".join([start_part, end_part])


def shapefile_to_csv(src_path, dest_path):
    pbar = trange(2, desc="Reading file", leave=False)
    data = gpd.read_file(src_path)
    pbar.update()
    pbar.set_description("Saving to output file")
    data.to_csv(dest_path, index=False)
    pbar.update()
    pbar.close()
    print("Complete")


def flatten_geom(gdf_src, id_column, geometry_column="geometry", leave=True):
    geometry = []
    rows = []

    pbar = trange(
        len(gdf_src.groupby(id_column)) + 1, desc="Flattening geometry", leave=leave
    )
    for _, group in gdf_src.groupby(id_column):
        polygon = shapely.unary_union(group.geometry)
        if isinstance(polygon, shapely.MultiPolygon):
            for poly in polygon.geoms:
                poly = normalize(poly)
                row = group.iloc[0].drop(geometry_column).to_dict()
                row["bbox"] = get_geom_bboxes(poly)[0]
                rows.append(row)
                geometry.append(poly)
        else:
            polygon = normalize(polygon)
            row = group.iloc[0].drop(geometry_column).to_dict()
            row["bbox"] = get_geom_bboxes(polygon)[0]
            rows.append(row)
            geometry.append(polygon)
        pbar.update()

    gdf = gpd.GeoDataFrame(rows, geometry=geometry, crs=gdf_src.crs)
    pbar.update()
    pbar.set_description("Flattening geometry. Complete")
    pbar.close()

    return gdf


def map_metadata(
    df_src, img_dir, parse_filename=parse_filename, leave=True
) -> pd.DataFrame:
    img_dir = Path(img_dir).resolve()
    columns = {
        "start_year": [],
        "end_year": [],
        "filename": [],
        "path": [],
        "width": [],
        "height": [],
        "bbox": [],
    }
    rows = []
    geometry = []

    pbar = trange(len(df_src) + 1, desc="Collecting image data", leave=leave)
    for _, row in df_src.iterrows():
        filename = parse_filename(row)
        path = img_dir / filename

        if path.exists():
            for r in rows:
                if path == r["path"]:
                    continue
            suffix = path.suffix
            open_fn = open_tif if suffix in [".tiff", ".tif"] else Image.open

            with open_fn(path) as img:
                width = str(img.shape[0])
                height = str(img.shape[1])
                rows.append(
                    {
                        "start_year": row["StartYear"],
                        "end_year": row["EndYear"],
                        "filename": filename,
                        "path": path,
                        "width": width,
                        "height": height,
                        "bbox": row["bbox"],
                    }
                )
                geometry.append(row["geometry"])
        pbar.update()

    pbar.set_description("Populating dataframe")

    df_dest = gpd.GeoDataFrame(rows, columns=columns, geometry=geometry, crs=df_src.crs)
    pbar.update()

    pbar.set_description("Collecting image data. Complete")
    pbar.close()

    return df_dest


def preprocess_shapefile(shpfile, id_column, img_dir, leave=True) -> gpd.GeoDataFrame:
    gdf = load_shapefile(shpfile)
    crs = gdf.crs
    gdf = flatten_geom(gdf, id_column=id_column, leave=leave)
    gdf = map_metadata(gdf, img_dir, leave=leave)
    return gpd.GeoDataFrame(gdf, crs=crs)


def get_geom_points(geom):
    match (geom.geom_type):
        case "Polygon":
            points = [point for point in normalize(geom.exterior.coords)]
        case "MultiPolygon":
            points = [
                [point for point in normalize(polygon.exterior.coords)]
                for polygon in geom.geoms
            ]
        case _:
            raise ValueError("Unknown geometry type")
    return points


def get_geom_bboxes(geom):
    boxes = []
    match (geom.geom_type):
        case "Polygon":
            boxes.append(normalize(shapely.box(*geom.bounds)))
        case "MultiPolygon":
            for geom in geom.geoms:
                boxes.append(normalize(shapely.box(*geom.bounds)))
        case _:
            print("Unknown geometry type")
    return boxes


def stringify_points(points):
    return " ".join([f"{point[0]} {point[1]}" for point in points])


def stringify_bbox(bbox):
    return f"{' '.join([str(x) for x in bbox])}"


def parse_points_list_str(s):
    points = []
    i = 0
    while i < len(s):
        if s[i] == "(":
            i += 1
            stop = s.index(")", i)
            point = s[i:stop].split(",")
            points.append((float(point[0]), float(point[1])))
            i = stop
        else:
            i += 1
    return points


def get_geom(df, *, geom_key="geometry", parse_key: FunctionType | None = None):
    geoms = []
    for _, row in df.iterrows():
        if parse_key is not None:
            geom = parse_key(row)
            if geom is not None:
                geoms.append(geom)
        else:
            geoms.append(row[geom_key])
    return geoms


def get_geom_polygons(geom, *, flatten=False):
    polygons = []

    match (geom.geom_type):
        case "Polygon":
            polygons.append(Polygon(geom))
        case "MultiPolygon":
            if not flatten:
                polygons.extend([Polygon(g) for g in list(geom.geoms)])
            else:
                flattened = []
                for polygon in [Polygon(p) for p in list(geom.geoms)]:
                    if len(flattened) == 0:
                        flattened.append(polygon)
                        continue
                    found = False
                    for i, flat in enumerate(flattened):
                        union = shapely.coverage_union(flat, polygon)
                        for poly in union.geoms:
                            # There is no union
                            if poly.equals(flat) or poly.equals(polygon):
                                continue
                            # Replace this polygon with the union
                            flattened[i] = poly
                            found = True
                            break
                        if found:
                            break
                    if not found:
                        flattened.append(polygon)
                polygons.extend(flattened)

        case _:
            raise ValueError("Unknown geometry type")
    return polygons


def normalize_shapefile(path):
    df_in = load_shapefile(path)
    columns = {key: [] for key in [*df_in.columns, "bbox"]}
    df_out = gpd.GeoDataFrame(columns=columns)
    df_out.drop("geometry", axis=1)

    pbar = trange(len(df_in), desc="Normalizing")
    for _, row in df_in.iterrows():
        row_out = dict(row)
        geom = row_out["geometry"]
        polygons = get_geom_polygons(geom)
        boxes = get_geom_bboxes(geom)
        for i, polygon in enumerate(polygons):
            row_out["bbox"] = [boxes[i]]
            row_out["geometry"] = [polygon]
            df_row = gpd.GeoDataFrame.from_dict(row_out)
            df_out = pd.concat([df_out, df_row])

        pbar.update()
    pbar.set_description("Complete")
    pbar.close()
    return df_out


def normalize_shapefile_with_metadata(shpfile, dir, filename_key=parse_filename):
    dir = Path(dir).resolve()
    df_out = normalize_shapefile(shpfile)
    columns = {
        "filename": [],
        "path": [],
        "width": [],
        "height": [],
    }

    pbar = trange(len(df_out) + 1, desc="Collecting image data")
    for _, row in df_out.iterrows():
        filename = filename_key(row)
        path = dir / filename
        try:
            tif = open_tif(path)
        except Exception:
            # Populate the columns with no data and continue
            columns["filename"].append("None")
            columns["path"].append("None")
            columns["width"].append("None")
            columns["height"].append("None")
            pbar.update()
            continue

        width = str(tif.shape[0])
        height = str(tif.shape[1])
        columns["filename"].append(filename)
        columns["path"].append(path)
        columns["width"].append(width)
        columns["height"].append(height)
        pbar.update()

    pbar.set_description("Populating dataframe")
    df_out = df_out.assign(**columns)
    pbar.update()
    pbar.set_description("Complete")
    pbar.close()

    return df_out


def default_one_hot_key(df_row):
    try:
        treatment = int(df_row["RetentionP"])
    except Exception:
        treatment = 2
    return ("1", "Treatment") if treatment == 1 else ("0", "NoTreatment")


def one_hot_encode(df: pd.DataFrame, key=default_one_hot_key):
    columns = {"class_id": [], "class_name": []}
    pbar = trange(len(df) + 1, desc="Encoding class data")

    for _, row in df.iterrows():
        id, name = key(row)
        columns["class_id"].append(id)
        columns["class_name"].append(name)
        pbar.update()
    df_encoded = df.copy()
    df_encoded.insert(0, "class_id", columns["class_id"])
    df_encoded.insert(1, "class_name", columns["class_name"])
    pbar.update()
    pbar.set_description("Complete")
    pbar.close()
    return df_encoded


def write_chip(data, *, transform, meta, output_path=None):
    meta.update(
        {
            "width": data.shape[0],
            "height": data.shape[1],
            "transform": transform,
        }
    )
    if output_path is None:
        # Create an in-memory Image
        chip = io.BytesIO()
        with rasterio.MemoryFile(chip) as mem:
            with mem.open(**meta) as dest:
                dest.write(data, 1)
            with mem.open() as src:
                arr = src.read(1)
                return arr.reshape(data.shape)
    else:
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(data, 1)  # 1 --> single-band


def create_chips_from_geotiff(
    geotiff_path, chip_size, crs, output_dir=None, exist_ok=False, leave=False
):
    geotiff_path = Path(geotiff_path)
    width = chip_size[0]
    height = chip_size[1]
    chips = []

    with rasterio.open(geotiff_path, crs=crs) as src:
        bounds = src.bounds
        rmin, cmin = src.index(bounds.left, bounds.top)
        rmax, cmax = src.index(bounds.right, bounds.bottom)
        rmin, rmax = min(rmin, rmax), max(rmin, rmax)
        cmin, cmax = min(cmin, cmax), max(cmin, cmax)

        if width is None or height is None:
            width = cmax - cmin
            height = rmax - rmin

        if width <= 0 or height <= 0:
            return []

        pbar = trange(
            rmin, rmax, height, desc=f"Processing {geotiff_path.name}", leave=leave
        )
        for row in range(rmin, rmax, height):
            for col in range(cmin, cmax, width):
                chip_output_path = None
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    chip_output_path = (
                        output_dir / f"{geotiff_path.stem}_chip_{col}_{row}.tif"
                    )
                    if chip_output_path.exists() and not exist_ok:
                        raise FileExistsError(
                            f"File '{chip_output_path}' aleady exists"
                        )
                    output_dir.mkdir(parents=True, exist_ok=True)

                chip_window = Window.from_slices(
                    rows=(row, row + height), cols=(col, col + width)
                )
                chip_data = src.read(1, window=chip_window)

                # does the image have relevant data?
                if (
                    chip_data.shape[0] == 0
                    or chip_data.shape[1] == 0
                    or chip_data.max() == 0
                    or chip_data.max() == src.nodata
                ):
                    continue

                chip_transform = src.window_transform(chip_window)
                chip = write_chip(
                    chip_data,
                    transform=chip_transform,
                    meta=src.meta.copy(),
                    output_path=chip_output_path,
                )
                chips.append((chip, src.xy(row, col)))
            pbar.update()
    pbar.close()

    return chips


def collect_filepaths(df: pd.DataFrame, column_name):
    return list(df.loc[:, column_name].values())


def preprocess_geotiff_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir,
    years=None,
    img_path_col="path",
    start_year_col="StartYear",
    end_year_col="EndYear",
    clean_dest=False,
):
    if years is None:

        def all_images(df):
            return df.loc[:, img_path_col].unique().tolist()

        get_filepaths = all_images
    else:

        def match_years(df):
            return (
                df.loc[
                    (df[start_year_col] == years[0]) & (df[end_year_col] == years[1]),
                    img_path_col,
                ]
                .unique()
                .tolist()
            )

        get_filepaths = match_years

    imgs = pathify(get_filepaths(gdf))

    if len(imgs) == 0:
        raise Exception("Could not find images")

    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    if clean_dest:
        pbar = trange(1, desc="Cleaning output directory", leave=False)
        clear_directory(output_dir)
        pbar.update()
        pbar.close()

    return imgs


def preprocess_ndvi_difference_geotiffs(
    gdf: gpd.GeoDataFrame,
    output_dir,
    *,
    years=None,
    img_path_col="path",
    start_year_col="start_year",
    end_year_col="end_year",
    geom_col="geometry",
    chip_size=None,
    exist_ok=False,
    clean_dest=False,
    ignore_empty_geom=True,
    leave=True,
):
    imgs = preprocess_geotiff_dataset(
        gdf,
        output_dir,
        years,
        img_path_col,
        start_year_col,
        end_year_col,
        clean_dest,
    )

    chip_size = chip_size if chip_size is not None else (None, None)
    if not isinstance(chip_size, tuple):
        chip_size = (chip_size, chip_size)

    pbar = trange(
        len(imgs) + 2,
        desc=f"Creating GeoTIFF chips of size {f'({chip_size[0]},{chip_size[1]})' if chip_size[0] is not None else 'Default'}",
        leave=leave,
    )

    with ThreadPoolExecutor(max_workers=get_cpu_count()) as executor:
        futures = [
            executor.submit(
                create_chips_from_geotiff,
                path,
                crs=gdf.crs,
                chip_size=chip_size,
                output_dir=output_dir / path.stem,
                exist_ok=exist_ok,
            )
            for path in imgs
        ]
        for _ in as_completed(futures):
            pbar.update()

    gdf = map_geometry_to_geotiffs(gdf, output_dir)
    pbar.update()

    chip_paths = [
        str(path)
        for path in collect_files_with_suffix(".tif", output_dir, recurse=True)
    ]
    unused_chips = []

    if ignore_empty_geom:
        # Remove any chips that do not map to an image in the dataframe
        unused_chips = gdf.loc[gdf[geom_col].is_empty, img_path_col].tolist()
        gdf = gdf[~gdf[geom_col].is_empty].reset_index(drop=True)

        for path in tqdm(unused_chips, desc="Cleaning up", leave=False):
            path = Path(path)
            parent = path.parent
            parent_parent = parent.parent
            if path.exists():
                os.remove(path)
            if parent.exists() and len(os.listdir(parent)) == 0:
                os.rmdir(parent)
                if parent_parent.exists() and len(os.listdir(parent_parent)) == 0:
                    os.rmdir(parent_parent)

    pbar.update()
    nfiles = max(0, len(chip_paths) - len(unused_chips))
    pbar.set_description(
        "Processed {0} images and saved to {1}".format(nfiles, output_dir)
    )
    pbar.close()

    return gdf.reset_index(drop=True)


def make_ndvi_dataset(
    shp_file,
    ndvi_dir,
    output_dir,
    *,
    years=None,
    id_column="Subregion",
    start_year_col="start_year",
    end_year_col="end_year",
    geom_col="geometry",
    chip_size=None,
    clean_dest=False,
    xy_to_index=True,
    exist_ok=False,
    save_csv=False,
    save_shp=False,
    ignore_empty_geom=True,
    tif_to_png=True,
    leave=True,
):
    if shp_file is None:
        raise ArgumentError("Missing path to shape file")
    if ndvi_dir is None:
        raise ArgumentError("Missing path to NDVI images")
    if output_dir is None:
        raise ArgumentError("Missing path to output directory")

    shp_file, ndvi_dir, output_dir = pathify(shp_file, ndvi_dir, output_dir)
    meta_dir = output_dir / "meta"
    csv_dir = meta_dir / "csv" / shp_file.stem
    shp_dir = meta_dir / "shp" / shp_file.stem

    if output_dir.exists() and clean_dest:
        clear_directory(output_dir)
    elif not output_dir.exists():
        output_dir.mkdir(parents=True)

    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=exist_ok)
    if save_shp:
        shp_dir.mkdir(parents=True, exist_ok=exist_ok)

    chips_dir = output_dir / "images" / "chips"
    chips_dir.mkdir(parents=True, exist_ok=exist_ok)

    n_calls = 4
    n_calls += 1 if xy_to_index else 0
    n_calls += 1 if tif_to_png else 0

    pbar = trange(
        n_calls, desc="Creating NDVI dataset - Preprocessing shapefile", leave=leave
    )

    gdf = preprocess_shapefile(
        shp_file, id_column=id_column, img_dir=ndvi_dir, leave=False
    )
    pbar.update()

    if years is not None:
        gdf = gdf.loc[
            (gdf[start_year_col] == years[0]) & (gdf[end_year_col] == years[1])
        ]

    output_fname = shp_file.stem
    if years is not None:
        output_fname += f"_{years[0]}to{years[1]}"
    output_fname = Path(output_fname)

    if save_csv:
        save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
    if save_shp:
        save_as_shp(
            gdf,
            shp_dir / output_fname.with_suffix(".shp"),
        )

    pbar.update()
    pbar.set_description("Creating NDVI dataset - Preprocessing GeoTIFFs")

    gdf = preprocess_ndvi_difference_geotiffs(
        gdf,
        chips_dir,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        geom_col=geom_col,
        years=years,
        chip_size=chip_size,
        clean_dest=clean_dest,
        exist_ok=exist_ok,
        ignore_empty_geom=ignore_empty_geom,
        leave=False,
    )
    pbar.update()

    if save_csv or save_shp:
        output_fname = Path(f"{output_fname}_chips_xy")
        if save_csv:
            save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / output_fname.with_suffix(".shp"),
            )

    if xy_to_index:
        pbar.update()
        pbar.set_description("Creating NDVI dataset - Translating xy coords to index")

        gdf = translate_xy_coords_to_index(gdf, leave=False)
        if save_csv or save_shp:
            output_fname = str(output_fname).replace("_xy", "_indexed")
            if not output_fname.endswith("_indexed"):
                output_fname += "_indexed"
            output_fname = Path(output_fname)
            if save_csv:
                save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
            if save_shp:
                save_as_shp(
                    gdf,
                    shp_dir / output_fname.with_suffix(".shp"),
                )
    if tif_to_png:
        pbar.update()
        pbar.set_description("Creating NDVI dataset - Converting GeoTIFFs to PNGs")

        tif_png_file_map = process_geotiff_to_png_conversion(
            chips_dir, output_dir / "images" / "png-chips", leave=False
        )
        spbar = trange(len(gdf), desc="Mapping filepaths", leave=False)
        for i, row in gdf.iterrows():
            tif_file = str(row["filename"])
            paths = tif_png_file_map.get(Path(tif_file).stem)
            if paths is not None:
                gdf.loc[i, "filename"] = paths["png"].name
                gdf.loc[i, "path"] = paths["png"]
            spbar.update()
        spbar.close()
        if save_csv or save_shp:
            output_fname = Path(f"{output_fname}_as_png")
            if save_csv:
                save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
            if save_shp:
                save_as_shp(
                    gdf,
                    shp_dir / output_fname.with_suffix(".shp"),
                )
    pbar.update()
    pbar.set_description("Creating NDVI dataset - Complete")
    pbar.close()

    return gdf, (meta_dir, chips_dir, output_fname)


def ndvi_to_yolo_dataset(
    shp_file,
    ndvi_dir,
    output_dir,
    *,
    years=None,
    id_column="Subregion",
    start_year_col="start_year",
    end_year_col="end_year",
    geom_col="geometry",
    chip_size=None,
    clean_dest=False,
    xy_to_index=True,
    class_parser=assign_default_classes,
    exist_ok=False,
    save_csv=False,
    save_shp=False,
    ignore_empty_geom=True,
    generate_labels=True,
    tif_to_png=True,
    use_segments=True,
    generate_train_data=True,
    split=0.75,
    split_mode="all",
    shuffle=True,
    background_bias=None,
):
    gdf, (meta_dir, chips_dir, output_fname) = make_ndvi_dataset(
        shp_file,
        ndvi_dir,
        output_dir,
        years=years,
        id_column=id_column,
        start_year_col=start_year_col,
        end_year_col=end_year_col,
        geom_col=geom_col,
        chip_size=chip_size,
        clean_dest=clean_dest,
        xy_to_index=xy_to_index,
        exist_ok=exist_ok,
        save_csv=save_csv,
        save_shp=save_shp,
        ignore_empty_geom=ignore_empty_geom,
        tif_to_png=tif_to_png,
        leave=False,
    )

    csv_dir = meta_dir / "csv"
    shp_dir = meta_dir / "shp"

    n_calls = 2
    n_calls += 1 if generate_labels else 0
    n_calls += 1 if generate_train_data else 0
    pbar = trange(n_calls, desc="Creating YOLO dataset - Encoding classes", leave=True)

    gdf = one_hot_encode(gdf, class_parser)
    if save_csv or save_shp:
        output_fname = Path(f"{output_fname}_encoded")
        if save_csv:
            save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / output_fname.with_suffix(".shp"),
            )

    labeled_images = gdf.loc[gdf["class_id"] != "-1"].values.tolist()

    if background_bias is None:
        gdf = gdf.loc[gdf["class_id"] == -1]
        new_rows = labeled_images
    else:
        background_images = gdf.loc[gdf["class_id"] == "-1"].values.tolist()[
            : len(labeled_images)
        ]
        new_rows = labeled_images + background_images

    gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)

    pbar.update()
    pbar.set_description("Creating YOLO dataset - Creating YOLODataset")

    yolo_ds = to_yolo(gdf)

    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    yolo_ds.generate_yaml_file(
        root_abs_path=output_dir,
        dest_abs_path=output_dir / "config",
        train_path=output_dir / "images" / "train",
        val_path=output_dir / "images" / "val",
        test_path=output_dir / "images" / "test",
    )

    train_data = None
    if generate_labels or generate_train_data:
        pbar.update()
        pbar.set_description("Creating YOLO dataset - Generating labels")

        yolo_ds.generate_label_files(
            output_dir / "labels" / "generated",
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
                output_dir / "images" / "png-chips" if tif_to_png else chips_dir
            )
            train_data = yolo_ds.split_data(
                ds_images_dir,
                output_dir / "labels" / "generated",
                split=split,
                shuffle=shuffle,
                recurse=True,
                mode=split_mode,
            )

            yolo_df = yolo_ds.data_frame
            yolo_ds.compile(get_cpu_count())
            yolo_ds.data_frame = yolo_df

    if save_csv:
        yolo_ds.to_csv(csv_dir / "yolo_ds.csv")

    pbar.update()
    pbar.set_description("Complete")
    pbar.close()
    return yolo_ds, train_data


def to_yolo(gdf: gpd.GeoDataFrame, compile=True) -> YOLODataset:
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: stringify_points(x.exterior.coords)
    )
    tmp_path = "/tmp/ftcnn_yolo_ds.csv"
    gdf.to_csv(tmp_path)
    try:
        ds = YOLODataset.from_csv(
            tmp_path,
            segments_key="geometry",
            convert_bounds_to_bbox=True,
            num_workers=get_cpu_count(),
            compile=compile,
        )
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    except Exception as e:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise e
    return ds


"""
    This might be the issue which causes the label issues where geometry 
    does not align with the actual geom,also causing the labeled geom to 
    appear "clipped" overlayed on the image
"""


def create_chip_polygon(src, chip_window):
    chip_transform = src.window_transform(chip_window)
    width, height = chip_window.width, chip_window.height

    chip_polygon = Polygon(
        [
            chip_transform * (0, 0),  # Top-left
            chip_transform * (width, 0),  # Top-right
            chip_transform * (width, height),  # Bottom-right
            chip_transform * (0, height),  # Bottom-left
            chip_transform * (0, 0),  # Close the polygon (back to top-left)
        ]
    )
    return chip_polygon


def map_geometry_to_geotiffs(
    gdf: gpd.GeoDataFrame, img_dir, recurse=True
) -> gpd.GeoDataFrame:
    img_dir = Path(img_dir).resolve()
    columns = [
        "parent",
        "filename",
        "path",
        "width",
        "height",
        "geometry",
    ]
    rows = []
    geometry = []

    for path in tqdm(
        collect_files_with_suffix(".tif", img_dir, recurse=recurse),
        desc="Mapping geometry to GeoTIFFs",
        leave=False,
    ):
        with rasterio.open(path) as src:
            # Maybe use Window.from_slices(rows=(0, n_rows), cols=(0, n_cols)), which is more explicit
            chip_window = Window.from_slices(rows=(0, src.height), cols=(0, src.width))
            chip_polygon = create_chip_polygon(src, chip_window)
            intersecting_polygons = gdf.loc[gdf.intersects(chip_polygon)]

            row = {
                "filename": path.name,
                "path": str(path),
                "width": src.width,
                "height": src.height,
            }

            if not intersecting_polygons.empty:
                for _, polygon_row in intersecting_polygons.iterrows():
                    geometry.append(polygon_row["geometry"].intersection(chip_polygon))
                    row["parent"] = str(polygon_row["path"])
                    rows.append(row)
            else:
                geometry.append(Polygon())
                row["parent"] = ""
                rows.append(row)

    return gpd.GeoDataFrame(
        rows, columns=columns, geometry=geometry, crs=gdf.crs
    ).explode()


def translate_xy_coords_to_index(gdf: gpd.GeoDataFrame, *, leave=True):
    gdf = gdf.copy()
    pbar = trange(len(gdf), desc="Translating geometry", leave=leave)
    for i, row in gdf.iterrows():
        if Path(str(row["path"])).exists() and isinstance(row["geometry"], Polygon):
            gdf.loc[i, "geometry"] = Polygon(
                geotiff_convert_geometry_to_pixels(row["path"], row["geometry"])
            )
        pbar.update()
    if leave:
        pbar.set_description("Complete")
    pbar.close()
    return gdf


def parse_polygon_str(polygon_str: str):
    size = len(polygon_str)
    start = 0
    end = size - 1
    while (
        start < end
        and start < size
        and end >= 0
        and not (polygon_str[start].isdigit() and polygon_str[end].isdigit())
    ):
        if not polygon_str[start].isdigit():
            start += 1
        if not polygon_str[end].isdigit():
            end -= 1

    if start < size and end >= 0:
        polygon_str = polygon_str[start : end + 1]
    parsed = []
    points = polygon_str.split(", ")
    for point in points:
        point = point.replace(" 0", "").replace("(", "").replace(")", "")
        x, y = point.split()
        parsed.append((float(x), float(y)))

    return parsed


def geotiff_convert_geometry_to_pixels(tiff_path, geometry):
    def __append_point_pixels(points, pixels, src):
        for point in points:
            pixels.append(
                src.index(point[0], point[1])[::-1],
            )

    if not isinstance(geometry, Polygon):
        if isinstance(geometry, str):
            geometry = parse_polygon_str(geometry)
        else:
            raise ValueError(f"Unknown type '{type(geometry)}'")
        geometry = Polygon(geometry)

    polygon = normalize(geometry.simplify(0.002, preserve_topology=True))

    pixels = []
    with rasterio.open(tiff_path) as src:
        geom_points = list(polygon.exterior.coords)
        __append_point_pixels(geom_points, pixels, src)
    return clip_points(pixels, (src.width, src.height))


def geotiff_convert_pixels_to_geometry(tiff_path, pixels):
    def __append_point_coords(points, coords, src):
        for point in points:
            coords.append(
                src.xy(point[1], point[0]),
            )

    if not isinstance(pixels, Polygon):
        if isinstance(pixels, list):
            if not isinstance(pixels[0], tuple):
                pixels = [(pixels[i], pixels[i + 1]) for i in range(len(pixels) - 1)]
        elif isinstance(pixels, str):
            pixels = parse_polygon_str(pixels)
        else:
            raise ValueError(f"Unknown type '{type(pixels)}'")
        pixels = normalize(Polygon(pixels))

    polygon = pixels.simplify(0.002, preserve_topology=True)

    coords = []
    with rasterio.open(tiff_path) as src:
        pixel_points = list(normalize(polygon.exterior.coords))
        __append_point_coords(pixel_points, coords, src)
    return Polygon(coords)


def clip_points(points, shape):
    width = shape[0]
    height = shape[1]
    clipped = []
    for x, y in points:
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        clipped.append((x, y))
    return clipped


def tiff_to_png(tiff, out_path=None):
    if isinstance(tiff, str):
        src_ds = gdal.Open(tiff)
    else:
        src_ds = gdal_array.OpenArray(tiff)

    band = src_ds.GetRasterBand(1)
    data = band.ReadAsArray()

    # Interpolate values from -1,1 to 0,1
    data = linterp(data, 0, 1)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize

    driver = gdal.GetDriverByName("MEM")
    rgb_ds = driver.Create("/tmp/png-from-tiff.tif", width, height, 3, gdal.GDT_Float64)

    for band in range(1, 4):
        rgb_ds.GetRasterBand(band).WriteArray(data)
        rgb_ds.GetRasterBand(band).SetNoDataValue(-1)
    rgb_ds.FlushCache()

    png_data = rgb_ds.ReadAsArray()
    png_data = linterp(png_data, 0, 1)
    # Left-rotate the array from 1, height, width to height, width, 1
    png_data = np.moveaxis(png_data, 0, -1)

    png_data = img_as_float(png_data)
    if out_path is not None:
        skio.imsave(out_path, (png_data * 255).astype(np.uint8), check_contrast=False)
    rgb_ds = None
    src_ds = None

    return (png_data * 255).astype(np.uint8)


def normalize_tiff(src_path, dest, *, vmin=None, vmax=None):
    src_ds = gdal.Open(src_path)

    translate_opts = ["-ot", "Byte", "-co", "TILED=YES", "-a_nodata", "0", "-scale"]

    # Translate and save the image
    dest_ds = gdal.Translate(dest, src_ds, options=" ".join(translate_opts))

    # Free the ds memory
    src_ds = None
    dest_ds = None


def process_geotiff_to_png_conversion(
    src_dir, dest_dir, *, recurse=True, preserve_dir=True, clear_dir=True, leave=True
):
    file_map = {}

    src_dir = Path(src_dir).absolute()
    dest_dir = Path(dest_dir).absolute()

    src_paths = collect_files_with_suffix(".tif", src_dir, recurse=recurse)
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    elif clear_dir:
        clear_directory(dest_dir)

    minval = sys.maxsize
    maxval = -sys.maxsize

    for path in src_paths:
        ds = gdal.Open(path)
        for i in range(1, ds.RasterCount):
            rb = ds.GetRasterBand(i)
            rb.SetNoDataValue(0)
            arr = rb.ReadAsArray()
            minval = min(minval, np.min(arr))
            maxval = max(maxval, np.max(arr))
        ds = None

    pbar = trange(len(src_paths), desc="Converting TIFF to PNG", leave=leave)
    for path in src_paths:
        if preserve_dir:
            relpath = Path(os.path.relpath(path, src_dir))
            dest_path = Path(dest_dir, relpath.with_suffix(".png"))
        else:
            dest_path = dest_dir / path.with_suffix(".png").name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        norm_path = "/tmp/norm.tif"
        normalize_tiff(path, norm_path, vmin=minval, vmax=maxval)
        tiff_to_png(norm_path, dest_path)
        file_map[path.stem] = {"tif": path, "png": dest_path}
        pbar.update()
    if leave:
        pbar.set_description("Complete")
    pbar.close()

    return file_map
