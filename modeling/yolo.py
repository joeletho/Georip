import os
import random
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep, time

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib import pyplot as plt
from pandas.compat import sys
from PIL import Image, ImageDraw
from shapely import MultiPolygon, Polygon, normalize, unary_union
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from tqdm.auto import tqdm, trange

from ftcnn.ftcnn import (chip_geotiff_and_convert_to_png, clear_directory,
                         encode_classes, get_cpu_count, make_ndvi_dataset,
                         save_as_csv, save_as_shp, stringify_points)

from .types import BBox, XYInt, YOLODataset
from .utils import (display_image_and_annotations,
                    extract_annotated_label_and_image_data, write_classes)

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

TQDM_INTERVAL = 1 / 100


class YOLODatasetViewer(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, anns_dir, transforms=None):
        """
        Args:
            imgs_dir (str): Directory where the images are stored.
            anns_dir (str): Directory where the YOLO annotation files are stored.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.imgs_dir = imgs_dir
        self.anns_dir = anns_dir
        self.transforms = transforms
        self.classes = ["treatment"]

        # Get all image files
        self.img_files = [
            file_name
            for file_name in os.listdir(imgs_dir)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.img_files.sort()

    def __getitem__(self, idx):
        # Get image file name and corresponding annotation file
        img_file = self.img_files[idx]
        img_path = os.path.join(self.imgs_dir, img_file)
        ann_path = os.path.join(self.anns_dir, os.path.splitext(img_file)[0] + ".txt")

        # Load the image
        img = read_image(img_path)
        img = tv_tensors.Image(img)

        # Get image dimensions
        img_height, img_width = F.get_size(img)  # returns (H, W)

        # Initialize lists for boxes, labels, masks
        boxes = []
        labels = []
        masks = []

        # Check if annotation file exists
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # Skip invalid lines

                    class_id = int(parts[0])

                    if len(parts) == 5:
                        # Standard YOLO format: class_id x_center y_center width height
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert from normalized coordinates to pixel coordinates
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height

                        # Convert from center coordinates to corner coordinates
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2

                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

                        # Create a mask from the bounding box
                        mask = torch.zeros((img_height, img_width), dtype=torch.uint8)
                        x_min_int = int(round(x_min))
                        y_min_int = int(round(y_min))
                        x_max_int = int(round(x_max))
                        y_max_int = int(round(y_max))
                        mask[y_min_int:y_max_int, x_min_int:x_max_int] = 1
                        masks.append(mask.numpy())
                    else:
                        # Assume the rest of the parts are polygon coordinates
                        # Format: class_id x1 y1 x2 y2 x3 y3 ... xn yn
                        coords = list(map(float, parts[1:]))
                        if len(coords) % 2 != 0:
                            continue  # Invalid polygon

                        x_coords = coords[::2]
                        y_coords = coords[1::2]

                        # Convert normalized coordinates to pixel coordinates
                        x_coords = [x * img_width for x in x_coords]
                        y_coords = [y * img_height for y in y_coords]

                        # Create a polygon
                        polygon = [(x, y) for x, y in zip(x_coords, y_coords)]

                        # Create a mask from the polygon
                        mask_img = Image.new("L", (img_width, img_height), 0)
                        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
                        mask = np.array(mask_img, dtype=np.uint8)
                        masks.append(mask)

                        # Compute bounding box
                        x_min = min(x_coords)
                        x_max = max(x_coords)
                        y_min = min(y_coords)
                        y_max = max(y_coords)
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

        else:
            # If annotation file doesn't exist, return empty annotations
            boxes = []
            labels = []
            masks = []

        # Convert to tensors
        if len(boxes) > 0:
            boxes = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img)
            )
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = tv_tensors.Mask(masks)
            # Compute area
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            boxes = tv_tensors.BoundingBoxes(
                [], format="XYXY", canvas_size=F.get_size(img)
            )
            labels = torch.as_tensor([], dtype=torch.int64)
            masks = tv_tensors.Mask([])
            area = torch.as_tensor([], dtype=torch.float32)
            iscrowd = torch.as_tensor([], dtype=torch.int64)

        image_id = idx

        # Prepare the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_files)

    def draw(
        self,
        show=True,
        save_dir=None,
        include_background=False,
        verbose=False,
        pbar=False,
        leave=False,
        pbar_desc="Drawing annotations",
    ):
        if pbar:
            total_updates = len(self)
            updates = 0
            start = time()
            pbar = trange(total_updates, desc=pbar_desc, leave=leave)
        for i in range(len(self)):
            display_image_and_annotations(
                self,
                idx=i,
                save_dir=save_dir,
                show=show,
                include_background=include_background,
                verbose=verbose,
            )
            if pbar and time() - start >= TQDM_INTERVAL:
                pbar.update()
                updates += 1
                start = time()
        if pbar:
            if updates < total_updates:
                pbar.update(total_updates - updates)
            pbar.close()


def plot_yolo_results(
    results, *, shape: XYInt | None = None, figsize: XYInt | None = None
):
    if shape is None:
        shape = (1, len(results))

    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    elif shape[0] == 1:
        axes = np.array(axes)
    axes = axes.ravel()

    if len(axes) < len(results):
        raise Exception(
            "Invalid shape: number of results exceeds the shape of the plot"
        )

    for i, r in enumerate(results):
        img = r.plot()
        axes[i].imshow(img)
    plt.show()


def yolo_make_dataset(labels_path, images_path, class_map, root_dir):
    root_dir = Path(root_dir).resolve()
    labels_dir = root_dir / "labels"
    images_dir = root_dir / "images"

    pbar = trange(3, desc="Preparing root directory", leave=False)

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    pbar.update()

    clear_directory(labels_dir)
    pbar.update()
    clear_directory(images_dir)
    pbar.update()
    pbar.close()

    for id in tqdm(class_map.keys(), desc="Copying to root directory", leave=False):
        yolo_copy_labels_and_images_containing_class(
            str(id),
            src_labels_dir=labels_path,
            src_images_dir=images_path,
            dest_dir=root_dir,
        )
    pbar = trange(1, desc="Preparing labels and classes")
    classes = [str(id) for id in list(class_map.keys())]
    print(classes)
    yolo_remove_annotations_not_in(classes, labels_dir=labels_dir)

    pbar.update()
    classes, _ = yolo_recategorize_classes(
        class_map,
        labels_dir,
    )
    write_classes(classes, root_dir / "classes.txt")
    pbar.set_description("Complete")
    pbar.close()


def yolo_get_labels_and_images(
    label_paths, image_paths, class_map, *, num_workers=None, from_format="yolo"
):
    labels = []
    images = []

    def __remove_not_in__(sources, targets):
        results = []
        for spath in tqdm(sources, desc="Cleaning unused sources", leave=False):
            for tpath in targets:
                if spath.stem == tpath.stem:
                    results.append(spath)
                    break
        return results

    def __preprocess_paths__(a, b):
        pbar = trange(2, desc="Preprocessing paths", leave=False)
        a = [Path(a, p) for p in os.listdir(a)]
        a.sort(key=lambda p: p.stem)
        b = [Path(b, p) for p in os.listdir(b)]
        b.sort(key=lambda p: p.stem)
        pbar.update()

        a = __remove_not_in__(a, b)
        b = __remove_not_in__(b, a)
        pbar.update()

        if len(a) != len(b):
            raise Exception(
                "Provided paths to not map. Each label path must have a associated image path"
            )
        pbar.close()
        return a, b

    def __collect_labels_and_images__(lpaths, ipaths, classes):
        lbls = []
        imgs = []
        if len(lpaths) != len(ipaths):
            raise Exception("Path lists must have the same length")

        for i in trange(len(lpaths), desc="Collecting labels and images", leave=False):
            if lpaths[i].stem != ipaths[i].stem:
                raise Exception(f"Path stems at index {i} do not match")

            extracted_labels, image = extract_annotated_label_and_image_data(
                lpaths[i], ipaths[i], classes
            )
            if from_format != "yolo":
                for label in extracted_labels:
                    bbox = label.bbox
                    bbox.x = (bbox.x + bbox.width) / 2
                    bbox.y = (bbox.y + bbox.height) / 2

            lbls.extend(extracted_labels)
            imgs.append(image)
            sleep(0.1)
        return lbls, imgs

    label_paths, image_paths = __preprocess_paths__(label_paths, image_paths)
    if num_workers is None:
        num_workers = get_cpu_count()
    batch = len(label_paths) // num_workers

    pbar = trange(num_workers, desc="Progress")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(label_paths), batch):
            futures.append(
                executor.submit(
                    __collect_labels_and_images__,
                    label_paths[i : i + batch],
                    image_paths[i : i + batch],
                    class_map,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            labels.extend(result[0])
            images.extend(result[1])
            pbar.update()

    pbar.set_description("Complete")
    pbar.close()

    return labels, images


def yolo_copy_labels_and_images_containing_class(
    class_id, *, src_labels_dir, src_images_dir, dest_dir
):
    label_paths = []
    image_paths = []
    labels_dest = Path(dest_dir, "labels").resolve()
    images_dest = Path(dest_dir, "images").resolve()
    class_id = str(class_id)

    for filename in tqdm(
        os.listdir(src_labels_dir), desc="Collecting label paths", leave=False
    ):
        label_path = Path(src_labels_dir, filename).resolve()
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) and parts[0] == class_id:
                    label_paths.append(label_path)
                    break
    labels = {p.stem: p for p in label_paths}
    for filename in tqdm(
        os.listdir(src_images_dir), desc="Collecting image paths", leave=False
    ):
        stem = os.path.splitext(filename)[0]
        if labels.get(stem):
            image_paths.append(Path(src_images_dir, filename))

    images = {p.stem: p for p in image_paths}
    for stem, p in labels.items():
        if not images.get(stem):
            label_paths.remove(p)

    for label in tqdm(label_paths, desc="Copying lables", leave=False):
        shutil.copy(label, labels_dest / label.name)

    for image in tqdm(image_paths, desc="Copying images", leave=False):
        shutil.copy(image, images_dest / image.name)
    print(f"Complete. Copied {len(label_paths)} labels and images")


def yolo_remove_annotations_not_in(class_ids, *, labels_dir):
    labels_dir = Path(labels_dir).resolve()
    files_annotations = {}
    filenames = os.listdir(labels_dir)
    for filename in tqdm(filenames, desc="Collecting class annotations"):
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        with open(path) as f:
            for line in f:
                if len(line) == 0:
                    continue
                parts = line.split()
                if parts[0] in class_ids:
                    if not files_annotations.get(filename):
                        files_annotations[filename] = []
                    files_annotations[filename].append(line)

    for filename in tqdm(filenames, desc="Writing to files"):
        annotations = files_annotations.get(filename)
        if not annotations:
            continue
        lines = "\n".join(line for line in annotations)
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        with open(path, "w") as f:
            f.write(lines)

    print(f"Complete. {len(files_annotations.keys())} files written")


def yolo_recategorize_classes(classes: dict, labels_dir):
    labels_dir = Path(labels_dir).resolve()
    old_new_map = {}
    for filename in tqdm(os.listdir(labels_dir), desc="Collecting class ids"):
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        with open(path) as f:
            for line in f:
                line.strip()
                parts = line.split()
                if len(parts) == 0:
                    continue
                id = str(parts[0])
                if id not in old_new_map.keys():
                    old_new_map[id] = None
    for i, id in enumerate(old_new_map.keys()):
        old_new_map[str(id)] = str(i)

    for filename in tqdm(os.listdir(labels_dir), desc="Writing to files"):
        path = labels_dir / filename
        if not path.is_file() or path.suffix != ".txt":
            continue
        lines = []
        with open(path, "r+") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 0:
                    continue
                id = str(parts[0])
                if id in [str(k) for k in old_new_map.keys()]:
                    parts[0] = old_new_map[id]
                line = " ".join(part for part in parts)
                if len(line) > 0:
                    lines.append(line)
            f.truncate(0)
            f.seek(0)
            f.write("{}\n".format("\n".join(line for line in lines)))

    print(old_new_map)
    new_classes = {}
    for old, new in old_new_map.items():
        name = classes.get(str(old))
        new_classes[str(new)] = name

    for name in new_classes.values():
        if name == "None":
            raise Exception(f"Class assignment failed: {new_classes}")

    print("Complete")

    return new_classes, old_new_map


def convert_xml_bbox_to_yolo(df: pd.DataFrame):
    pbar = tqdm(
        total=df.shape[0], desc="Converting XML BBox to YOLO format", leave=False
    )
    for _, row in df.iterrows():
        bbox = BBox(
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        )

        bbox.width -= bbox.x
        bbox.height -= bbox.y

        row["bbox_x"] = bbox.x
        row["bbox_y"] = bbox.y
        row["bbox_w"] = bbox.width
        row["bbox_h"] = bbox.height
        pbar.update()
    pbar.close()


def convert_xml_dataframe_to_yolo(df: pd.DataFrame):
    df.rename(
        columns={
            "filename": "filename",
            "name": "class_name",
            "width": "width",
            "height": "height",
            "xmin": "bbox_x",
            "ymin": "bbox_y",
            "xmax": "bbox_w",
            "ymax": "bbox_h",
        },
        inplace=True,
    )


def predict_on_image_stream(model, *, images, conf=0.6, **kwargs):
    batch_size = kwargs.get("batch_size")
    if batch_size is None:
        num_workers = kwargs.get("num_workers")
        batch_size = num_workers if num_workers is not None else get_cpu_count()
    else:
        kwargs.pop("batch_size")
    for i in range(0, len(images) - 1, batch_size):
        try:
            results = model.predict(
                source=[image[0] for image in images[i : i + batch_size]],
                conf=conf,
                stream=True,
                verbose=False,
                **kwargs,
            )
            for j, result in enumerate(results):
                yield get_result_stats(result), images[i + j][1]
        except Exception as e:
            print(e, file=sys.stderr)
            yield None


def predict_on_image(model, image, conf=0.6):
    result = model(image, conf=conf)[0]
    return get_result_stats(result)


def get_result_stats(result):
    # Detection
    classes = result.boxes.cls.cpu().numpy()  # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()  # box with xyxy format, (N, 4)

    # Segmentation
    if result.masks is None:
        masks = None
    else:
        masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)

    return result, (boxes, masks, classes, probs)


def get_input_size(model):
    info = model.named_parameters()
    return list(info)[0][1].shape[0]


def mask_to_polygon(mask):
    # Assuming mask is binary, extract polygons from mask
    mask = (mask * 255).astype(np.uint8).squeeze()
    # print(type(mask), mask.shape, mask.dtype, mask.min(), mask.max())
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = [Polygon(c.reshape(-1, 2)) for c in contours if len(c) >= 3]
    return polygons


def predict_geotiff(model, geotiff_path, confidence, chip_size, imgsz, **kwargs):
    chips, epsg_code = chip_geotiff_and_convert_to_png(
        geotiff_path, chip_size=chip_size
    )
    results = []

    pbar = tqdm(total=len(chips), desc="Detections 0", leave=False)
    for result in predict_on_image_stream(
        model, imgsz=imgsz, images=chips, conf=confidence, **kwargs
    ):
        if result is not None and result[0][1][1] is not None:
            results.append(result)
            pbar.set_description(f"Detections {len(results)}")
        pbar.update()
    pbar.update()
    pbar.close()

    columns = [
        "path",
        "class_id",
        "class_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "geometry",
    ]
    rows = []
    geometry = []

    with rasterio.open(geotiff_path) as src:
        for (result, data), coords in results:
            row, col = src.index(*coords)
            for mask in data[1]:  # Assuming this is the segmentation mask
                polygons = mask_to_polygon(mask)

                class_id = data[2][0]
                class_name = result.names[class_id]
                for bbox in data[0]:
                    bbox_x, bbox_y, bbox_xx, bbox_yy = (
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                    )
                    bbox_y, bbox_x = src.xy(row + bbox_y, col + bbox_x)
                    bbox_yy, bbox_xx = src.xy(row + bbox_yy, col + bbox_xx)
                    bbox_w = bbox_xx - bbox_x
                    bbox_h = bbox_yy - bbox_y
                    for unioned_geometry in polygons:
                        unioned_geometry = Polygon(
                            [
                                src.xy(row + y, col + x)
                                for x, y in unioned_geometry.exterior.coords
                            ]
                        )
                        rows.append(
                            {
                                "path": geotiff_path,
                                "class_id": int(class_id),
                                "class_name": class_name,
                                "bbox_x": bbox_x,
                                "bbox_y": bbox_y,
                                "bbox_w": bbox_w,
                                "bbox_h": bbox_h,
                            }
                        )
                        geometry.append(unioned_geometry.buffer(0))

    gdf = gpd.GeoDataFrame(
        rows, columns=columns, geometry=geometry, crs=f"EPSG:{epsg_code}"
    )

    if gdf.empty:
        return results, gdf

    # Union all intersecting polygons
    unioned_geometry = unary_union(gdf["geometry"])

    distinct_geometries = []
    if isinstance(unioned_geometry, MultiPolygon):
        for poly in unioned_geometry.geoms:
            poly = normalize(poly)
            distinct_geometries.append(poly)
    else:
        unioned_geometry = normalize(unioned_geometry)
        geometry.append(unioned_geometry)

    # If unioned_geometry is a MultiPolygon, each separate polygon will be handled as distinct
    #
    # if isinstance(unioned_geometry, Polygon):
    #     distinct_geometries = [unioned_geometry]  # A single unioned Polygon
    # elif isinstance(unioned_geometry, MultiPolygon):
    #     distinct_geometries = list(
    #         unioned_geometry.geoms
    #     )  # List of separate polygons in the unioned result

    rows = []
    geometry = []
    for geom in distinct_geometries:
        # Find rows in the original gdf that match this unionized geometry
        matching_rows = gdf[
            gdf["geometry"].intersects(geom)
        ]  # Find all original rows that intersect this new geometry

        # Add a new row for the unioned geometry, keeping other relevant information from the first matching row
        if not matching_rows.empty:
            row_to_keep = matching_rows.iloc[
                0
            ].copy()  # Copy the first matching row to keep its other fields
            # Update the geometry to the unionized one
            rows.append(row_to_keep)
            geometry.append(geom)
    gdf_unionized = gpd.GeoDataFrame(
        rows, columns=columns, geometry=geometry, crs=gdf.crs
    )

    return results, gdf_unionized


def predict_geotiffs(
    model, geotiff_paths, *, confidence, chip_size, imgsz, max_images=2, **kwargs
):
    results = []
    gdfs = []

    def get_index_with_crs(gdf):
        for i, g in enumerate(gdfs):
            if g.crs == gdf.crs:
                return i
        return -1

    pbar = trange(len(geotiff_paths), desc="Processing predictions", leave=False)

    with ThreadPoolExecutor(max_workers=max_images) as executor:
        futures = [
            executor.submit(
                predict_geotiff,
                model,
                path,
                confidence,
                chip_size=chip_size,
                imgsz=imgsz,
                **kwargs,
            )
            for path in geotiff_paths
        ]
        for future in as_completed(futures):
            if future.exception() is not None:
                print(future.exception(), file=sys.stderr)
            else:
                result, _gdf = future.result()
                results.append(result)
                if len(gdfs) == 0:
                    gdfs.append(_gdf)
                elif not _gdf.empty:
                    index = get_index_with_crs(_gdf)
                    if index == -1:
                        gdfs.append(_gdf)
                    else:
                        gdfs[index] = pd.concat([gdfs[index], _gdf], ignore_index=True)
            pbar.update()
    pbar.close()

    return results, gdfs


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
    encoder=encode_classes,
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
    shuffle_split=True,
    shuffle_background=True,
    background_bias=None,
    min_labels_required=10,
    pbar_leave=True,
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
        ignore_empty_geom=ignore_empty_geom and background_bias is None,
        tif_to_png=tif_to_png,
        pbar_leave=False,
    )

    csv_dir = meta_dir / "csv"
    shp_dir = meta_dir / "shp"

    n_calls = 3
    n_calls += 1 if generate_labels else 0
    n_calls += 1 if generate_train_data else 0
    pbar = trange(n_calls, desc="Creating YOLO dataset - Encoding classes", leave=pbar_leave)

    gdf = encode_classes(gdf, encoder)
    if save_csv or save_shp:
        output_fname = Path(f"{output_fname}_encoded")
        if save_csv:
            save_as_csv(gdf, csv_dir / output_fname.with_suffix(".csv"))
        if save_shp:
            save_as_shp(
                gdf,
                shp_dir / output_fname.with_suffix(".shp"),
            )

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

    gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)
    pbar.update()

    if len(gdf) < min_labels_required:
        pbar.set_description("Failed to create YOLO dataset - Error occured")
        pbar.close()
        raise ValueError(
            f"Minimum number of labels is less than the minimum required: got {len(gdf)} when at least {min_labels_required} are required."
        )

    pbar.set_description(
        f"Creating YOLO dataset - Creating YOLODataset with {len(gdf)} labels"
    )
    yolo_ds = to_yolo(gdf)
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
                output_dir / "images" / "png-chips" if tif_to_png else chips_dir
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
