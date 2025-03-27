# Georip: Geospacial Raster Image Processing

Georip is a repository designed for processing geospatial and raster data to be used in convolutional neural networks (CNNs). This project focuses on raster manipulation, allowing users to create datasets from geospatial data (such as satellite images or aerial imagery) that can be used for CNN-based image classification and segmentation. The repository provides Jupyter notebooks for dataset creation, model prediction, and visualization of results, with an emphasis on forest management and environmental monitoring applications.

## Table of Contents

- [Description](#description)
- [Setup](#setup)
  - [Setting up the Environment](#setting-up-the-environment)
  - [Activating the Environment](#activating-the-environment)
- [Usage](#usage)
  - [Dataset Creation](#dataset-creation)
  - [Prediction](#prediction)
  - [Viewing Images and Labels](#viewing-images-and-labels)
- [License](#license)

## Description

Georip is specifically designed for processing geospatial raster data (such as satellite images, lidar data, and aerial imagery). The primary goal of this project is to apply deep learning models, particularly convolutional neural networks (CNNs), to classify and analyze raster data for forest management and environmental monitoring. The project enables users to manipulate and preprocess raster data into training datasets for CNN models, helping to automate the classification of forest conditions, land use changes, and other relevant environmental phenomena.

Key features include:

- Raster data processing tailored for forest treatment (FT) applications.
- Data augmentation techniques designed for geospatial images.
- Model training and fine-tuning using CNNs for geospatial classification.
- Jupyter notebooks to guide users through dataset creation, model training, prediction, and visualization.

## Setup

### Setting up the Environment

To set up your environment, this repository includes an `environment.yml` file that lists all the necessary dependencies.

1. Clone the repository:

   ```bash
   git clone https://github.com/joeletho/Georip.git
   cd Georip
   ```
   
2. Create the Conda environment using the provided environment.yml:

  ```bash
  conda env create -f conda/environment.yml
  ```

This will create a Conda environment with all the required dependencies for running the project.

### Activating the Environment

To activate the environment, run:
  ```bash
  conda activate georip
  ```

Once activated, you can start using the provided Jupyter notebooks for dataset creation, model training, prediction, and result visualization.

## Usage

### Dataset Creation

[Create a single dataset](https://github.com/joeletho/Georip/blob/main/notebooks/yolo_create_single_dataset.ipynb), or [create multiple datasets](https://github.com/joeletho/Georip/blob/main/notebooks/yolo_create_multiple_datasets.ipynb) with different configurations (useful for testing dataset performance and bias).

The dataset creation process involves manipulating geospatial raster data to create labeled datasets for training CNNs. The provided notebooks guide you through tasks like:

- Loading geospatial raster data (e.g., satellite imagery or other forest-related raster datasets).
- Preprocessing raster images, including resizing, normalization, and dividing them into tiles suitable for CNN input.
- Labeling data based on forest treatment categories, either manually or through automated methods.
- Augmenting the dataset by applying transformations such as rotation, flipping, and scaling to improve model robustness.

### Train Models

Georip integrates both [Ultralytics YOLO](https://docs.ultralytics.com/) and [MaskRCNN](https://github.com/matterport/Mask_RCNN) models with tools for dataset creation, training, and more.

### Prediction

Easily [predict on multiple](https://github.com/joeletho/Georip/blob/main/notebooks/yolo_predict_multiple_datasets.ipynb) validation by storing or linking the dataset folders in a single directory. This may be useful when developing a dataset for your model(s). [Create multiple datasets](https://github.com/joeletho/Georip/blob/main/notebooks/yolo_create_multiple_datasets.ipynb) with different configurations (train/val/test splits, image sizes, etc.) and then immediately run predictions to obtain performance metrics.

After training the CNN model, you can use the prediction notebook to classify new geospatial raster data. This notebook allows you to:

- Load a trained model.
- Make predictions on new raster data (e.g., classify new satellite or forest images).
- Visualize the predictions overlaid on the original images to assess accuracy and identify areas of interest.

### Viewing Images and Labels

[View datasets](https://github.com/joeletho/Georip/blob/main/notebooks/yolo_dataview.ipynb) or a subset, as a pair of images and/or annotations. This is especially useful to visually validate your data (i.e. all image operations are sound and annotation pixels map to the correct pixels), so that when you start training, you can be assured the data is correct.

To view images and their associated labels, open the relevant notebook in the notebooks/ folder. These notebooks provide a way to:

- Display geospatial raster images alongside their labels for validation.
- Compare predicted labels with the actual ground truth labels.
- Inspect any misclassifications and analyze how well the model performs in forest treatment and environmental contexts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

