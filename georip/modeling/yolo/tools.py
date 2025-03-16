import os
from pathlib import Path

from cv2.typing import MatLike
from PIL import Image
from skimage import io
from supervision.annotators.core import cv2
from tqdm.auto import tqdm, trange

from georip.io import clear_directory, collect_files_with_suffix
from georip.modeling.utils import write_classes
from georip.modeling.yolo.utils import (
    copy_labels_and_images_containing_class,
    recategorize_classes,
    remove_annotations_not_in,
)
from georip.utils import StrPathLike


def create_truth_and_prediction_pairs(
    truth_images_dir: StrPathLike,
    truth_labels_dir: StrPathLike,
    pred_images_dir: StrPathLike,
) -> list[tuple[Image, Image]]:
    """
    Creates pairs of ground truth and predicted images with YOLO bounding boxes for evaluation.

    Parameters:
        truth_images_dir (str): Directory containing the ground truth images.
        truth_labels_dir (str): Directory containing the ground truth YOLO label files.
        pred_images_dir (str): Directory containing the predicted images.

    Returns:
        list of tuple: A list of pairs of images (ground truth and predicted images).

    Example:
        images = yolo_create_truth_and_prediction_pairs("truth_images", "truth_labels", "pred_images")
    """
    truth_image_paths = collect_files_with_suffix([".jpg", ".png"], truth_images_dir)
    truth_label_paths = collect_files_with_suffix(".txt", truth_labels_dir)
    pred_image_paths = collect_files_with_suffix([".jpg", ".png"], pred_images_dir)

    assert (
        len(truth_image_paths) == len(truth_label_paths)
        and "Number of Images and labels must match"
    )
    assert (
        len(truth_image_paths) == len(pred_image_paths)
        and "Number of truth images must match predicted images"
    )

    # Align image and label paths
    for i in range(len(truth_image_paths)):
        found = False
        for j, label in enumerate(truth_label_paths):
            if truth_image_paths[i].stem == label.stem:
                truth_label_paths[i], truth_label_paths[j] = (
                    truth_label_paths[j],
                    truth_label_paths[i],
                )
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Could not find label for {truth_image_paths[i].name}"
            )

    # Create truth-bounded images and add it and its predicted counterpart to the list of images
    images = []
    for i in range(len(truth_image_paths)):
        found = False
        for pred_path in pred_image_paths:
            if truth_image_paths[i].stem == pred_path.stem:
                truth_image = draw_yolo_bboxes(
                    truth_image_paths[i], truth_label_paths[i]
                )
                pred_image = io.imread(pred_path)
                images.append((truth_image, pred_image))
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Could not find {truth_image_paths[i].name} in predicted images"
            )

    return images


def make_dataset(
    labels_path: StrPathLike,
    images_path: StrPathLike,
    class_map: dict[str | int, str],
    root_dir: StrPathLike,
) -> None:
    """
    Prepares a YOLO dataset by copying relevant labels and images into a structured directory.

    Parameters:
        labels_path: StrPathLike
            Path to the directory containing label files.
        images_path: StrPathLike
            Path to the directory containing image files.
        class_map: Dict[int, str]
            Mapping of class IDs to class names.
        root_dir: StrPathLike
            Path to the root directory where the YOLO dataset will be created.

    Returns:
        None

    Raises:
        Exception: If copying labels and images fails.
    """
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
        copy_labels_and_images_containing_class(
            str(id),
            src_labels_dir=labels_path,
            src_images_dir=images_path,
            dest_dir=root_dir,
        )
    pbar = trange(1, desc="Preparing labels and classes")
    classes = [str(id) for id in list(class_map.keys())]
    remove_annotations_not_in(classes, labels_dir=labels_dir)

    pbar.update()
    classes, _ = recategorize_classes(
        class_map,
        labels_dir,
    )
    write_classes(classes, root_dir / "classes.txt")
    pbar.set_description("Complete")
    pbar.close()


def draw_yolo_bboxes(
    image_path: StrPathLike, label_path: StrPathLike, class_names: None | list[str] = None
) -> MatLike:
    """
    Draws YOLO-style bounding boxes on an image based on the provided label file.

    Parameters:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO label file (text file containing bounding boxes).
        class_names (list of str, optional): A list of class names for labeling the bounding boxes. Defaults to None.

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.

    Example:
        img_with_bboxes = draw_yolo_bboxes("image.jpg", "image.txt", class_names=["class1", "class2"])
    """
    # Load the image using OpenCV
    img = cv2.imread(str(image_path))
    img_height, img_width = img.shape[:2]

    # Read the YOLO-formatted label file
    with open(label_path, "r") as f:
        bboxes = f.readlines()

    # Loop through each line (bounding box) in the label file
    for bbox in bboxes:
        bbox = bbox.strip().split()

        class_id = int(bbox[0])  # Class ID is the first value
        x_center = float(bbox[1])  # YOLO X center (relative to image width)
        y_center = float(bbox[2])  # YOLO Y center (relative to image height)
        bbox_width = float(bbox[3])  # YOLO width (relative to image width)
        bbox_height = float(bbox[4])  # YOLO height (relative to image height)

        # Convert YOLO coordinates back to absolute pixel values
        x_center_abs = int(x_center * img_width)
        y_center_abs = int(y_center * img_height)
        bbox_width_abs = int(bbox_width * img_width)
        bbox_height_abs = int(bbox_height * img_height)

        # Calculate the top-left corner of the bounding box
        x_min = int(x_center_abs - (bbox_width_abs / 2))
        y_min = int(y_center_abs - (bbox_height_abs / 2))
        x_max = int(x_center_abs + (bbox_width_abs / 2))
        y_max = int(y_center_abs + (bbox_height_abs / 2))

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Bounding box color (green)
        thickness = 2  # Thickness of the box
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Optionally, label the bounding box with the class name
        if class_names:
            label = class_names[class_id]
            cv2.putText(
                img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # Convert the image back to RGB for display with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb
