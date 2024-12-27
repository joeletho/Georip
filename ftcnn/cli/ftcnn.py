import argparse
from importlib import import_module, util
from importlib.abc import InspectLoader
from pathlib import Path

import src.modeling.ftcnn.ftcnn as ftcnn

"""
def ndvi_to_yolo_dataset(
    shp_file,
    ndvi_dir,
    output_dir,
    *,
    years=None,
    start_year_col="StartYear",
    end_year_col="EndYear",
    geom_col="geometry",
    chip_size=(None, None),
    clean_dest=False,
    xy_to_index=True,
    class_parser,
    exist_ok=False,
    save_csv=False,
    save_shp=False,
    ignore_empty_geom=True,
    generate_labels=True,
    tif_to_png=True,
    use_segments=True,
    generate_train_data=True,
    split=0.7,
    shuffle=True,
):
"""


def import_module_from_path(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"No module found at {module_path}")

    module_name = module_path.stem
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--shape_file", type=Path, required=True)
    parser.add_argument("-n", "--ndvi_dir", type=Path, required=True)
    parser.add_argument("-o", "--out_dir", type=Path, required=True)
    parser.add_argument("-y", "--years", type=int, nargs=2, default=None)
    parser.add_argument("-z", "--chip_size", type=int, nargs=2, default=[None, None])
    parser.add_argument("-r", "--clean_dest", action="store_true", default=False)
    parser.add_argument("-t", "--translate_xy", action="store_false", default=True)
    parser.add_argument("-e", "--exist_ok", action="store_true", default=False)
    parser.add_argument("-c", "--save_csv", action="store_true", default=False)
    parser.add_argument("-a", "--save_shp", action="store_true", default=False)
    parser.add_argument("-i", "--ignore_empty_geom", action="store_true", default=False)
    parser.add_argument("-l", "--generate_labels", action="store_false", default=True)
    parser.add_argument("-g", "--convert_to_png", action="store_false", default=True)
    parser.add_argument("-u", "--use_segments", action="store_false", default=True)
    parser.add_argument(
        "-d", "--generate_train_data", action="store_false", default=True
    )
    parser.add_argument("-v", "--split", type=float, default=0.7)
    parser.add_argument("-s", "--shuffle", action="store_false", default=True)

    msg = 'def classify(row):\n\
        geom = row.get("geometry")\n\
        return (\n\
        ("0", "Treatment")\n\
        if geom is not None and not geom.is_empty and geom.area > 1\n\
        else ("-1", "Background")\n\
        )'
    parser.add_argument(
        "-p",
        "--class_parser",
        type=str,
        nargs=2,
        metavar="<path/to/module>.<method>",
        help=f"Callback method used to label the classes in the dataset. For example, you would pass `--class_parser path/to/utils classify` to use the function 'classify' in 'utils.py' module (utils.classify).\n  Example of possible parser method:\n    {msg}",
        default=True,
        required=True,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    parser_module = args.class_parser[0]

    try:
        parser_module = import_module(parser_module)
    except Exception:
        parser_module = import_module_from_path(Path(parser_module))

    parser_method = args.class_parser[1]
    class_parser = getattr(parser_module, parser_method)

    ds, data = ftcnn.ndvi_to_yolo_dataset(
        args.shape_file,
        args.ndvi_dir,
        args.out_dir,
        years=(None if args.years is None else (args.years[0], args.years[1])),
        chip_size=(args.chip_size[0], args.chip_size[1]),
        clean_dest=args.clean_dest,
        class_parser=class_parser,
        xy_to_index=args.translate_xy,
        exist_ok=args.exist_ok,
        save_csv=args.save_csv,
        save_shp=args.save_shp,
        ignore_empty_geom=args.ignore_empty_geom,
        generate_labels=args.generate_labels,
        tif_to_png=args.convert_to_png,
        use_segments=args.use_segments,
        generate_train_data=args.generate_train_data,
        split=args.split,
        shuffle=args.shuffle,
    )

    ds.to_csv(out_dir / "yolo_ndvi_ds.csv")

    if data is not None:
        with open(out_dir / "train_images.txt", "w") as f:
            for p in data[0]:
                f.write(f"{p}\n")
        with open(out_dir / "train_labels.txt", "w") as f:
            for p in data[1]:
                f.write(f"{p}\n")
        with open(out_dir / "val_images.txt", "w") as f:
            for p in data[2]:
                f.write(f"{p}\n")
        with open(out_dir / "val_labels.txt", "w") as f:
            for p in data[3]:
                f.write(f"{p}\n")
