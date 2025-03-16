import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import yaml

from georip.geoprocessing import DataFrameLike
from georip.utils import StrPathLike

from .geoprocessing import load_geo_dataframe, load_shapefile, open_geo_dataset

__all__ = [
    "save_as_csv",
    "save_as_shp",
    "save_as_gpkg",
    "load_shapefile",
    "load_geo_dataframe",
    "open_geo_dataset",
    "save_as_yaml",
]


def pathify(path: StrPathLike | list[StrPathLike], *args) -> Path | list[Path]:
    """
    Convert input paths into `Path` objects.

    Parameters:
        path (StrPathLike | list[StrPathLike]): A single path or list of paths to be converted.
        *args (StrPathLike): Additional paths to append to the result.

    Returns:
        Path | list[Path]:
            - If a single path is provided, returns a single `Path` object.
            - If a list of paths is provided, returns a list of `Path` objects.

    Example:
        >>> pathify("folder/file.txt")
        PosixPath('folder/file.txt')

        >>> pathify(["file1.txt", "file2.txt"], "file3.txt")
        [PosixPath('file1.txt'), PosixPath('file2.txt'), PosixPath('file3.txt')]
    """
    is_array = isinstance(path, list)
    if not is_array:
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]
    if len(args):
        paths.extend([Path(arg) for arg in args])
    return paths if isinstance(path, list) else paths[0]


def clear_directory(dir_path: StrPathLike) -> None:
    """
    Remove all files and subdirectories within a directory, preserving the directory itself.

    Parameters:
        dir_path (StrPathLike): Path to the directory to be cleared.

    Returns:
        None

    Notes:
        - Deletes all files and subdirectories within `dir_path`.
        - Preserves the `dir_path` directory itself.

    Example:
        >>> clear_directory("output_dir")
        # "output_dir" is now an empty directory.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"The path '{dir_path}' is not a directory.")

    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def collect_files_with_suffix(
    suffix: str | list[str], dir_path: StrPathLike, *, recurse: bool = False
) -> list[Path]:
    """
    Collect files with a specific suffix or list of suffixes in a directory.

    Parameters:
        suffix (str | list[str]): File suffix (e.g., ".txt") or a list of suffixes to filter files.
        dir_path (StrPathLike): Path to the directory to search for files.
        recurse (bool, optional): Whether to search subdirectories recursively. Defaults to False.

    Returns:
        list[Path]: A list of `Path` objects for files matching the suffix(es).

    Example:
        >>> collect_files_with_suffix(".txt", "input_dir")
        [PosixPath('input_dir/file1.txt'), PosixPath('input_dir/file2.txt')]

        >>> collect_files_with_suffix([".py", ".md"], "src", recurse=True)
        [PosixPath('src/main.py'), PosixPath('src/docs/README.md')]
    """
    dir_path = Path(dir_path)
    files = []
    if not isinstance(suffix, list):
        suffix = [suffix]
    for path in dir_path.iterdir():
        if path.is_dir() and recurse:
            files.extend(collect_files_with_suffix(suffix, path, recurse=recurse))
        else:
            if path.suffix in suffix:
                files.append(path)
    return files


def save_as_shp(
    gdf: gpd.GeoDataFrame, path: StrPathLike, exist_ok: bool = False
) -> None:
    """
    Save a GeoDataFrame as an ESRI Shapefile (.shp).

    Parameters:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to save.
        path (StrPathLike): The file path to save the Shapefile.
        exist_ok (bool, optional): If False, raises a FileExistsError if the file exists.
                                   Defaults to False.

    Raises:
        FileExistsError: If the file exists and `exist_ok` is False.

    Notes:
        - Automatically creates parent directories if they do not exist.
        - Shapefile format requires multiple files (.shp, .shx, .dbf, etc.), so `path`
          should exclude file extensions.

    Example:
        >>> save_as_shp(gdf, "output_folder/data.shp")
    """
    path = Path(path)
    if not exist_ok and path.exists():
        raise FileExistsError(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(path, driver="ESRI Shapefile")


def save_as_gpkg(
    gdf: gpd.GeoDataFrame, path: StrPathLike, exist_ok: bool = False
) -> None:
    """
    Save a GeoDataFrame as a GeoPackage (.gpkg).

    Parameters:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to save.
        path (StrPathLike): The file path to save the GeoPackage.
        exist_ok (bool, optional): If False, raises a FileExistsError if the file exists.
                                   Defaults to False.

    Raises:
        FileExistsError: If the file exists and `exist_ok` is False.

    Notes:
        - Automatically creates parent directories if they do not exist.
        - GeoPackage is a modern and versatile file format that supports multiple layers.

    Example:
        >>> save_as_gpkg(gdf, "output_folder/geo_data.gpkg")
    """
    path = Path(path)
    if not exist_ok and path.exists():
        raise FileExistsError(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(path, driver="GPKG")


def save_as_csv(df: DataFrameLike, path: StrPathLike, exist_ok: bool = False) -> None:
    """
    Save a DataFrame or GeoDataFrame as a CSV file.

    Parameters:
        df (DataFrameLike): The DataFrame or GeoDataFrame to save.
        path (StrPathLike): The file path to save the CSV file.
        exist_ok (bool, optional): If False, raises a FileExistsError if the file exists.
                                   Defaults to False.

    Raises:
        FileExistsError: If the file exists and `exist_ok` is False.

    Notes:
        - Automatically creates parent directories if they do not exist.
        - Geometry columns in a GeoDataFrame will be ignored when saving as CSV.

    Example:
        >>> save_as_csv(df, "output_folder/data.csv")
    """
    path = Path(path)
    if not exist_ok and path.exists():
        raise FileExistsError(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


def save_as_yaml(
    data: dict | argparse.Namespace,
    filepath: StrPathLike,
    *,
    mode: str = "w+",
    verbose: bool = True,
    parents: bool = False,
    exist_ok: bool = False,
    message: str | None = None,
):
    """
    Saves a dictionary to a YAML file, converting Path objects to strings.

    Args:
        data (dict | argparse.Namespace): The dictionary or Namespace to save.
        filename (StrPathLike): The output YAML file path, including filename.
        mode (str): The file mode for writing to the file.
        verbose (bool): The flag to print the completion message.
        parents (bool): The flag to create parent directories if they do not already exist.
        exist_ok (bool): The flag to allow overwriting an already-existing file.
    """
    filepath = Path(filepath).resolve()

    if isinstance(data, argparse.Namespace):
        data = vars(data)

    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: convert_paths(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    cleaned_data = convert_paths(data)

    if not filepath.exists():
        filepath.parent.mkdir(parents=parents, exist_ok=exist_ok)

    if not exist_ok:
        raise FileExistsError(f"{str(filepath)} already exists")

    with open(filepath, mode) as f:
        yaml.dump(cleaned_data, f, default_flow_style=False)

    if verbose:
        if message is None:
            print(f"Dictionary saved to {str(filepath)}")
        else:
            print(message)
