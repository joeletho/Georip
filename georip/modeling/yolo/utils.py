import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from ultralytics.engine.results import Results

from georip.modeling.utils import XYInt, extract_annotated_label_and_image_data
from georip.utils import NUM_CPU, StrPathLike


def get_labels_and_images(
    label_paths: list[StrPathLike],
    image_paths: list[StrPathLike],
    class_map: dict[str | int, str],
    *,
    num_workers: None | int = None,
):
    """
    Retrieves labels and images associated with specified classes.

    Parameters:
        label_paths: StrPathLike
            Directory containing label files.
        image_paths: StrPathLike
            Directory containing image files.
        class_map: Dict[int, str]
            Mapping of class IDs to class names.
        num_workers: int, optional
            Number of worker threads to use. Defaults to the number of CPUs.

    Returns:
        Tuple[List[Labels], List[Images]]
            Lists of labels and images.

    Raises:
        Exception: If label and image paths do not align or if any label fails to load.
    """
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
        num_workers = NUM_CPU
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


def copy_labels_and_images_containing_class(
    class_id: str | int,
    *,
    src_labels_dir: StrPathLike,
    src_images_dir: StrPathLike,
    dest_dir: StrPathLike,
) -> None:
    """
    Copies labels and images containing a specific class to a destination directory.

    Parameters:
        class_id: str
            The class ID to filter and copy.
        src_labels_dir: StrPathLike
            Source directory containing label files.
        src_images_dir: StrPathLike
            Source directory containing image files.
        dest_dir: StrPathLike
            Destination directory to store filtered labels and images.

    Returns:
        None

    Raises:
        IOError: If copying files fails.
    """
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


def remove_annotations_not_in(class_ids: list[str | int], *, labels_dir: StrPathLike):
    """
    Removes annotations in label files that do not match specified class IDs.

    Parameters:
        class_ids: List[str]
            A list of valid class IDs to retain.
        labels_dir: StrPathLike
            Directory containing label files.

    Returns:
        None
    """
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


def recategorize_classes(
    classes: dict[str | int, str], labels_dir: StrPathLike
) -> tuple[dict[str | int, str], dict[str | int, str]]:
    """
    Recategorizes class IDs in label files and maps old IDs to new IDs.

    Parameters:
        classes: Dict[str, str]
            Dictionary mapping old class IDs to new class names.
        labels_dir: StrPathLike
            Directory containing label files.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]
            Updated class dictionary and mapping of old to new IDs.

    Raises:
        Exception: If recategorization fails.
    """
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

    new_classes = {}
    for old, new in old_new_map.items():
        name = classes.get(str(old))
        new_classes[str(new)] = name

    for name in new_classes.values():
        if name == "None":
            raise Exception(f"Class assignment failed: {new_classes}")

    print("Complete")

    return new_classes, old_new_map


def get_result_stats(
    result: Results,
) -> tuple[Results, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts detection and segmentation statistics from YOLO results.

    Parameters:
        result: YOLOResult
            The result object from YOLO model prediction.

    Returns:
        Tuple[YOLOResult, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            Processed detection and segmentation results.
    """
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


def plot_results(
    results: list[Results], *, shape: XYInt | None = None, figsize: XYInt | None = None
) -> None:
    """
    Plots YOLO results in a grid layout.

    Parameters:
        results: List[YOLOResult]
            A list of YOLO result objects containing image and detection data to be plotted.
        shape: Tuple[int, int], optional
            The shape (rows, columns) of the plot grid. Defaults to (1, len(results)).
        figsize: Tuple[int, int], optional
            The size of the figure in inches (width, height). Defaults to None.

    Returns:
        None

    Raises:
        Exception: If the number of results exceeds the grid shape.
    """
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
