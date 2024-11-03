import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
from ftcnn.utils import clear_directory, get_cpu_count
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange

from .types import BBox, XYInt
from .utils import extract_annotated_label_and_image_data, write_classes

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)


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
    batch_size = get_cpu_count()
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
        except Exception as _:
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
