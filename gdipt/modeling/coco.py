import json
from pathlib import Path

import pandas as pd
from tqdm.auto import trange

from .utils import extrapolate_annotations_from_label


def coco_extrapolate_annotations(
    label_paths: list[str], img_paths: list[str], class_map: dict[int, str]
):
    UPDATE_INTERVAL = 3

    labels = []
    errors = []

    n_paths = len(label_paths)
    pbar = trange(n_paths, desc="Extrapolating annotations", leave=False)
    for i, label_path in enumerate(label_paths):
        interval = min(UPDATE_INTERVAL, n_paths - i)
        label = Path(label_path).resolve()
        if label.suffix == ".txt":
            for image_path in img_paths:
                image = Path(image_path)
                if image.stem == label.stem:
                    extrapolated, err = extrapolate_annotations_from_label(
                        label, image, class_map
                    )
                    labels.extend(extrapolated)
                    if len(errors):
                        errors.append(err)
                    break
        if i % interval == 0:
            pbar.update(interval)
    pbar.close()
    return labels, errors


def coco_parse_json_categories(jsonpath):
    parsed = []
    data = None
    with open(jsonpath) as f:
        data = json.load(f)
    if data is None:
        raise Exception(f"Error loading JSON {jsonpath}")

    for category in data["categories"]:
        parsed.append(
            {
                "super_class": category["supercategory"],
                "class_id": int(category["id"]),
                "class_name": category["name"],
            }
        )

    return parsed


def coco_json_categories_to_csv(jsonpath, dest):
    jsonpath = Path(jsonpath).resolve()
    dest = Path(dest).resolve()
    if not dest.is_file():
        filename = "categories.csv"
        i = 1
        while Path(dest, filename).exists():
            filename = f"categories-{i}.csv"
            i += 1
        dest = dest / filename
    elif dest.suffix != ".csv":
        dest = dest / (dest.name + ".csv")

    categories = coco_parse_json_categories(jsonpath)
    df = pd.DataFrame(categories)
    df.to_csv(dest)


def coco_get_images_data(jsonpath):
    data = None
    with open(jsonpath) as f:
        data = json.load(f)
    if data is None:
        raise Exception(f"Error loading JSON {jsonpath}")

    images = {}
    for image in data["images"]:
        images[image["id"]] = {
            "filename": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "id": image["id"],
        }
    return images


def coco_get_category_map(jsonpath):
    categories = {}
    data = {}
    with open(jsonpath) as f:
        data = json.load(f)
    for category in data["categories"]:
        categories[int(category["id"])] = category
    return categories


def coco_annotations_to_dataframe(jsonpath):
    jsonpath = Path(jsonpath).resolve()
    """
    "images": [
        "file_name": "0000001.jpg",
        "height": 427,
        "width": 640
        "id": 12345
    ],
    "annotations": [
        {
            "image_id": 12345,
            "bbox": [
                123.4, # x
                123.4, # y
                56.78, # width
                56.78  # height
            ],
            "category_id": 12,
            "id": 6789
        },
        {...},
    ]
    """
    pbar = trange(2, position=0, desc="Preprocessing annotation JSON...", leave=False)
    images_data = coco_get_images_data(jsonpath)
    pbar.update()

    category_map = coco_get_category_map(jsonpath)
    pbar.update()
    pbar.close()

    columns = [
        "class_id",
        "class_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "image_filename",
        "image_width",
        "image_height",
    ]

    rows = []

    data = {}
    with open(jsonpath) as f:
        data = json.load(f)

    pbar = trange(len(data.keys()) + 6, desc="Building Dataframe")
    for annotation in data.get("annotations"):
        image_data = images_data[annotation.get("image_id")]
        bbox = annotation.get("bbox")

        rows.append(
            {
                "class_id": annotation.get("category_id"),
                "class_name": category_map[annotation.get("category_id")].get("name"),
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_width": bbox[2],
                "bbox_height": bbox[3],
                "image_filename": image_data.get("filename"),
                "image_width": image_data.get("width"),
                "image_height": image_data.get("height"),
            }
        )
        pbar.update()

    df = pd.DataFrame(rows, columns=columns).sort_values(by=["class_id"])

    pbar.update(5)
    pbar.close()

    return df


def coco_json_to_dataframe(
    jsonpath, *, classes: list[str] | None = None, parallel=True
):
    pbar = trange(1, position=0, desc="Building COCO Dataframe", leave=False)

    df = coco_annotations_to_dataframe(jsonpath)
    pbar.update()
    pbar.close()

    if classes is not None:
        pbar = trange(df.shape[0] + 10, desc="Dropping unused classes", leave=False)
        indices = []
        for i, row in df.iterrows():
            if row.get("class_name") not in classes:
                indices.append(i)
            pbar.update()
        df.drop(indices, inplace=True)
        pbar.update(10)
        pbar.close()

    return df
