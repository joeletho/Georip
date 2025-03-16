import argparse
import ast
import sys
from importlib import import_module, util
from importlib.abc import InspectLoader
from pathlib import Path

import yaml

dirpath = Path(__file__).parent.resolve()
if str(dirpath) in sys.path:
    sys.path.remove(str(dirpath))
package_path = str(dirpath.parent.parent)
if package_path not in sys.path:
    sys.path.append(package_path)

from georip.datasets import YOLONDVIDifferenceDataset

DEFUALT_CONFIG_PATH = dirpath / "default_config.yaml"


def import_module_from_path(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"No module found at {module_path}")

    module_name = module_path.stem
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_preserve_fields(fields):
    preserve_fields = []

    # Split the value by commas to handle multiple items
    items = [field.strip() for field in fields.split(",")]

    for item in items:
        try:
            parsed_item = ast.literal_eval(item)
            if isinstance(parsed_item, dict):
                preserve_fields.append(parsed_item)
            else:
                preserve_fields.append(item)
        except (ValueError, SyntaxError):
            preserve_fields.append(item)

    return preserve_fields


def parse_function_path(function_path: str):
    """
    Parses the function path string in the form 'path/to/module.func' into
    the module and function name.

    Args:
        function_path (str): Full path string in the form 'module_path.function_name'.

    Returns:
        tuple: A tuple containing the module path and the function name.
    """
    module_path, func_name = function_path.rsplit(".", 1)

    return type("FunctionPath", (object,), {"path": module_path, "name": func_name})


def parse_pathlike(pathlike):
    path = Path(pathlike)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def parse_tile_shape(parser, arg, *args):
    if arg is None:
        return [None, None]
    if len(args) > 0:
        if len(args) > 1:
            tuples = parser._get_option_tuples("--tile_shape")
            if not len(tuples):
                raise ValueError("Option tuples is empty")
            argument = tuples[0][0]
            raise argparse.ArgumentError(
                argument,
                f"Too many arguments. Expected 2 integers for tile width and height, instead got {len(args)+1}",
            )
        return [int(arg), int(args[0])]
    val = int(arg)
    return [val, val]


def parse_input_file(args, pathlike):
    path = parse_pathlike(pathlike)
    match (path.suffix):
        case ".shp":
            setattr(args, "shapefile", path)
        case ".yaml":
            setattr(args, "config", path)
        case _:
            raise RuntimeError(
                f"Invalid file type '{pathlike}'. Must either be a .yaml or .shp file type"
            )


def setup_parser():
    parser = argparse.ArgumentParser(
        description="georip CLI for creating geospatial dataset for YOLO"
    )
    parser.add_argument(
        "source",
        type=parse_pathlike,
        help="Path to a YAML data file containing all configuration parameters or the input shapefile (.shp) containing data",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="Path to the parent directory containing the source images",
    )
    parser.add_argument(
        "--output_dir",
        type=parse_pathlike,
        help="Path to the destination directory where the dataset will be saved",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="The target start and end year range to used compute the NDVI difference between two NDVI images, where start == (end - 1). Pass 'None' to compute differences for each consecutive year.",
    )
    parser.add_argument(
        "--tile_shape",
        type=lambda x: parse_tile_shape(parser, x),
        default=[None, None],
        help="The side 'length' or 'width,height' of the images to be tiled",
    )
    parser.add_argument("--region_column", type=str, help="Column for regions")
    parser.add_argument("--year_start_column", type=str, help="Column for start year")
    parser.add_argument("--year_end_column", type=str, help="Column for end year")
    parser.add_argument(
        "--class_column", type=str, help="Column containing the class names"
    )
    parser.add_argument(
        "--class_names",
        type=lambda names: [name for name in names.split(",")],
        help="A single or comma-separated list of the dataset class names",
    )

    # Optional arguments
    parser.add_argument(
        "--background_shapefile",
        type=parse_pathlike,
        default=None,
        help="Path to a shapefile (.shp) containing background data. The data will be processed and merged with the source shapefile before being passed to the program.",
    )
    parser.add_argument("--geometry_column", type=str, help="Column for geometry")

    parser.add_argument("--background_ratio", type=float, help="Background ratio")

    parser.add_argument(
        "--background_filter",
        help="Path to background filter callback function in the form 'path/to/module.function_name'",
        default=None,
        type=lambda value: parse_function_path(value) if value else None,
    )

    parser.add_argument(
        "--background_seed", type=int, help="Seed for background generation"
    )
    parser.add_argument(
        "--split_mode", type=str, choices=["all", "collection"], help="Split mode"
    )
    parser.add_argument("--train_split_ratio", type=float, help="Training split ratio")
    parser.add_argument("--test_split_ratio", type=float, help="Test split ratio")
    parser.add_argument(
        "--shuffle_split", type=bool, help="Whether to shuffle the split"
    )
    parser.add_argument("--shuffle_seed", type=int, help="Seed for shuffle split")
    parser.add_argument("--stratify", type=bool, help="Whether to stratify the split")
    parser.add_argument("--generate_labels", type=bool, help="Generate labels")
    parser.add_argument(
        "--generate_train_data", type=bool, help="Generate training data"
    )
    parser.add_argument("--tile_size", type=str, help="Tile size for images")
    parser.add_argument("--stride", type=str, help="Stride for tiling")
    parser.add_argument(
        "--translate_xy", type=bool, help="Whether to apply xy translation"
    )

    parser.add_argument(
        "--class_encoder",
        help="Path to class encoder function in the form 'path/to/module.function_name'",
        default=None,
        type=lambda value: parse_function_path(value) if value else None,
    )

    parser.add_argument(
        "--exist_ok", type=bool, help="Whether to allow existing output"
    )
    parser.add_argument(
        "--clear_output_dir", type=bool, help="Whether to clear the output directory"
    )
    parser.add_argument("--save_shp", type=bool, help="Whether to save the shapefile")
    parser.add_argument(
        "--save_gpkg", type=bool, help="Whether to save in geopackage format"
    )
    parser.add_argument("--save_csv", type=bool, help="Whether to save as CSV")
    parser.add_argument("--pbar_leave", type=bool, help="Whether to leave progress bar")
    parser.add_argument(
        "--convert_to_png", type=bool, help="Whether to convert to PNG format"
    )
    parser.add_argument("--use_segments", type=bool, help="Whether to use segments")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for parallel processing"
    )

    # Handle --preserve_fields to allow comma-separated values or dictionary for renaming
    parser.add_argument(
        "--preserve_fields",
        type=parse_preserve_fields,
        help="Fields to preserve (comma-separated or dictionary for renaming)",
    )

    return parser


def import_function(module_path, function_name):
    try:
        module = import_module(module_path)
    except Exception:
        module = import_module_from_path(module_path)

    return getattr(module, function_name)


def parse_args(parser):
    args = parser.parse_args()
    parse_input_file(args, args.source)

    # If a YAML data file is provided, load the configurations from the file
    if args.config:
        if not args.config.suffix == ".yaml":
            print("Error: The provided data file must be a YAML file")
            return
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Override all arguments with the ones from the data file
        for key, value in config.items():
            setattr(args, key, value)
    else:
        # If no data file is provided, make certain required arguments are passed in the CLI
        required_args = [
            "shapefile",
            "image_dir",
            "output_dir",
            "region_column",
            "year_start_column",
            "year_end_column",
            "class_column",
            "class_names",
        ]
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]

        if missing_args:
            print(f"Error: Missing required arguments: {', '.join(missing_args)}")
            return

        # Load configuration from a default YAML file if available
        print("No data file provided, using default configurations")
        config = load_config(DEFUALT_CONFIG_PATH)

        # Update arguments with values from YAML config if not provided via CLI
        for key, value in config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    if args.background_filter:
        if not isinstance(args.background_filter, bool):
            setattr(
                args,
                "background_filter",
                import_function(
                    args.background_filter.path, args.background_filter.name
                ),
            )
    if args.class_encoder:
        setattr(
            args,
            "class_encoder",
            import_function(args.class_encoder.path, args.class_encoder.name),
        )

    return args


def main():
    parser = setup_parser()
    args = parse_args(parser)

    YOLONDVIDifferenceDataset.create(
        source=args.shapefile,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        region_column=args.region_column,
        year_start_column=args.year_start_column,
        year_end_column=args.year_end_column,
        class_column=args.class_column,
        class_names=args.class_names,
        geometry_column=args.geometry_column,
        years=args.years,
        background_shapefile=args.background_shapefile,
        background_ratio=args.background_ratio,
        background_filter=args.background_filter,
        background_seed=args.background_seed,
        split_mode=args.split_mode,
        train_split_ratio=args.train_split_ratio,
        test_split_ratio=args.test_split_ratio,
        stratify=args.stratify,
        shuffle_split=args.shuffle_split,
        shuffle_seed=args.shuffle_seed,
        generate_labels=args.generate_labels,
        generate_train_data=args.generate_train_data,
        tile_size=args.tile_size,
        stride=args.stride,
        translate_xy=args.translate_xy,
        class_encoder=args.class_encoder,
        exist_ok=args.exist_ok,
        clear_output_dir=args.clear_output_dir,
        save_shp=args.save_shp,
        save_gpkg=args.save_gpkg,
        save_csv=args.save_csv,
        pbar_leave=args.pbar_leave,
        convert_to_png=args.convert_to_png,
        use_segments=args.use_segments,
        num_workers=args.num_workers,
        preserve_fields=args.preserve_fields,
    )


if __name__ == "__main__":
    main()
