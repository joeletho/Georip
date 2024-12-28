from os import PathLike
from pathlib import Path

from ftcnn.io import clear_directory


def init_dataset_filepaths(
    *,
    source_shp: PathLike,
    images_dir: PathLike,
    output_dir: PathLike,
    save_csv: bool = True,
    save_shp: bool = True,
    save_gpkg: bool = True,
    clean_dest: bool,
    exist_ok: bool = False
) -> dict[str, Path]:
    source_shp, images_dir, output_dir = (
        Path(source_shp),
        Path(images_dir),
        Path(output_dir),
    )
    meta_dir: Path = output_dir / "meta"
    csv_dir: Path = meta_dir / "csv" / source_shp.stem
    shp_dir: Path = meta_dir / "shp" / source_shp.stem

    if output_dir.exists() and clean_dest:
        clear_directory(output_dir)
    elif not output_dir.exists():
        output_dir.mkdir(parents=True)

    if save_csv:
        csv_dir.mkdir(parents=True, exist_ok=exist_ok)
    if save_shp or save_gpkg:
        shp_dir.mkdir(parents=True, exist_ok=exist_ok)

    tiles_dir: Path = output_dir / "images" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=exist_ok)
    return {
        "source_shp": source_shp,
        "images_dir": images_dir,
        "output_dir": output_dir,
        "meta_dir": meta_dir,
        "csv_dir": csv_dir,
        "shp_dir": shp_dir,
        "tiles_dir": tiles_dir,
    }
