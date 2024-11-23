import argparse
import os
from pathlib import Path

import pandas as pd

"""
1. Open file log.out
2. rfind('Validating') to find the last instance of "Validating" in the file
5. Skip two "\n" which will place the pointer on the results line
6. Parse column names: "Class", "Images", "Instances", "box(p",    "r", "map50", "map50-95)", "mask(p",    "r", "map50", "map50-95)"
7. Parse row values:     "all",      26,          13, 	0.428,  0.692,     0.5,       0.317,    0.428,  0.692,     0.5,       0.314
8. Append rows to dataframe:

         |              yaml             |  |                                               dataset                                                            |  |                             metrics                                     |
Columns: "model", "batch", "freeze", "iou", "treatment", "start_year", "end_year", "imgsz", "split", "mode", "shuffle-split", "background", "shuffle-background", Box(P",    "R", "mAP50", "mAP50-95)", "Mask(P",    "R", "mAP50", "mAP50-95)"

project/train/args.yaml to obtain train cfg:
1. Open args.yaml
2. find("task:") for task type
3. find("model:") for model used
4. find("batch:") for batch size
5. find("project:") for dataset conf:
	a. Parse model cfg from filename: rfind('train'), step back 1 to find the '/' (end). rfind('/', end) to find the next '/' (start). Then s[start:end] is the model conf.
	b. split('_'): 
		[0]: 'yolo', 
		[1]: 'treatment=TYPE'
		[2]: 'years=STARTtoEND'
		[3]: 'imgsz=IMGSZ'
		[4]: 'split=XX'
		[5]: 'mode=MODE'
		[6]: 'shuffle-split=BOOL'
		[7]: 'bg=None' | 'bg=N'			        # if bg=N, then [8] should also contain a number, so the remaining places will increment by 1
		[8]: 'shuffle-bg=BOOL'
6. find("freeze:") for frozen layers 			# if 'null' then no layers were frozen
7. find("iou:") for IoU thresh

"""


def parse_metrics(path):
    data = {}
    line = ""
    with open(path, "rb") as f:
        train_images = None
        train_bg = None
        train_corrupt = None
        val_images = None
        val_bg = None
        val_corrupt = None

        for line in f:
            line = line.decode("utf-8")
            if (
                not train_images
                and "train:" in line
                or not val_images
                and "val:" in line
            ):
                parts = line.split()
                if len(parts) < 9:
                    continue

                indices = []
                for i, part in enumerate(parts):
                    if "train:" in part or "val:" in part:
                        indices.append(i)

                last = indices[-1]
                imgs = int(parts[last + 3])
                bgs = int(parts[last + 5])
                corr = int(parts[last + 7])

                if "train:" in parts[last]:
                    train_images = imgs
                    train_bg = bgs
                    train_corrupt = corr
                else:
                    val_images = imgs
                    val_bg = bgs
                    val_corrupt = corr

            if train_images and val_images:
                break

        if train_images is None or train_bg is None or train_corrupt is None:
            raise ValueError(
                f"Error finding parsing image data in '{path.name}': missing train image data"
            )
        if val_images is None or val_bg is None or val_corrupt is None:
            raise ValueError(
                f"Error finding parsing image data in '{path.name}': missing val image data"
            )
        f.seek(0, os.SEEK_END)
        size = f.tell()
        curr = size
        found = False

        while curr > 0:
            # Move pointer backward step-by-step to read in reverse
            curr -= 1
            f.seek(curr)
            char = f.read(1).decode("utf-8", errors="replace")

            if char == "\n" and curr < size - 1:  # Skip the trailing newline
                # Read the line after finding a newline
                f.seek(curr + 1)
                line = f.readline().decode("utf-8").strip().split()
                if len(line) > 0 and line[0] == "Validating":
                    found = True
                    break

        if not found:
            raise ValueError(
                f"Error finding 'Validating': unexpectedly reached end of file '{path.name}'"
            )
        n_lines = 0
        while f.tell() < size and n_lines < 3:
            if f.read(1).decode("utf-8", errors="replace") == "\n":
                n_lines += 1
        if n_lines != 3:
            raise ValueError(
                f"Error finding metrics values: unexpectantly reached end of file '{path.name}'"
            )

        metrics = (
            f.readline().decode("utf-8", errors="replace").strip("\n").strip().split()
        )
        if len(metrics) != 11:
            raise ValueError(
                f"Error parsing metrics: expected 11 fields, got {len(metrics)} in '{path.name}'"
            )

        data = {
            "images": [train_images + val_images],
            "backgrounds": [train_bg + val_bg],
            "corrupt": [train_corrupt + val_corrupt],
            "train_images": [train_images],
            "train_background": [train_bg],
            "train_corrupt": [train_corrupt],
            "val_images": [val_images],
            "val_background": [val_bg],
            "val_corrupt": [val_corrupt],
            "box_p": [metrics[3]],
            "box_r": [metrics[4]],
            "box_map50": [metrics[5]],
            "box_map50-95)": [metrics[6]],
            "mask_p": [metrics[7]],
            "mask_r": [metrics[8]],
            "mask_map50": [metrics[9]],
            "mask_map50-95)": [metrics[10]],
        }

    if len(data) == 0:
        raise ValueError(f"Error parsing metrics in '{path.name}': missing data")

    return pd.DataFrame.from_dict(data)


def parse_stats(path):
    data = {
        "project_name": [],
        "treatment": [],
        "start_year": [],
        "end_year": [],
        "imgsz": [],
        "split": [],
        "mode": [],
        "shuffle_split": [],
        "background_factor": [],
        "shuffle_background": [],
        "model": [],
        "batch": [],
        "freeze": [],
        "iou": [],
    }
    lines = None
    with open(path) as f:
        lines = f.readlines()
    if lines is None:
        raise ValueError(f"Error reading lines from file '{path.name}'")
    for i, line in enumerate(lines):
        pair = line.strip("\n").split(":")
        if len(pair) != 2:
            raise ValueError(
                f"Error reading key value pair in '{path.name}': malformed string on line {i}"
            )
        key = pair[0].strip()
        val = pair[1].strip()

        match (key):
            case "project":
                last_slash = val.rfind("/")
                project_name = val[last_slash:].strip('/')
                parts = project_name.split("_")
                n_fields = len(parts)
                if not (9 <= n_fields <= 10):
                    print(parts)
                    raise ValueError(
                        f"Error parsing project in '{path.name}': unrecognized format on line {i}"
                    )

                data["project_name"].append(project_name)

                # Parse parts
                """
                [1]: 'treatment=TYPE'
                [2]: 'years=STARTtoEND'
                [3]: 'imgsz=IMGSZ'
                [4]: 'split=XX'
                [5]: 'mode=MODE'
                [6]: 'shuffle-split=BOOL'
                [7]: 'bg=None' | 'bg=N'			        # if bg=N, then [8] should also contain a number, so the remaining places will increment by 1
                [8]: 'shuffle-bg=BOOL'
                """
                data["treatment"].append(parts[1].split("=")[1])

                years = parts[2].split("=")[1]
                if years == "None":
                    years = ["None", "None"]
                else:
                    years = years.split("to")
                    if len(years) != 2:
                        raise ValueError(
                            f"Error parsing project in '{path.name}': unrecognized 'years' format on line {i}"
                        )
                data["start_year"].append(
                    int(years[0]) if years[0].isnumeric() else years[0]
                )
                data["end_year"].append(
                    int(years[1]) if years[1].isnumeric() else years[1]
                )

                data["imgsz"].append(int(parts[3].split("=")[1]))
                data["split"].append(int(parts[4].split("=")[1]) / 100)
                data["mode"].append(parts[5].split("=")[1])
                data["shuffle_split"].append(bool(parts[6].split("=")[1]))

                data["background_factor"].append(parts[7].split("=")[1])
                next = 8
                if not parts[next].isnumeric():
                    if data["background_factor"][-1].isnumeric():
                        data["background_factor"][-1] = int(data["background"])
                    else:
                        data["background_factor"][-1] = "None"
                else:
                    data["background_factor"][-1] = ".".join(
                        [data["background_factor"][-1], parts[next]]
                    )

                    next += 1
                data["shuffle_background"].append(bool(parts[next].split("=")[1]))
            case _:
                # We don't want duplicates of what is parsed from 'project'
                if key not in ["mode", "split", "imgsz"] and key in data.keys():
                    digits = "".join(val.split("."))
                    if digits.lstrip("-").isnumeric():
                        if "." in val:
                            data[key].append(float(val))
                        else:
                            data[key].append(int(val))
                    else:
                        data[key].append(val)
    if not isinstance(data["freeze"][-1], int):
        data["freeze"][-1] = "None"

    return pd.DataFrame.from_dict(data)


def get_yaml_and_log_files(dir, args):
    yaml_file = dir / args.target_yaml_file
    log_file = dir / args.target_log_file

    if not yaml_file.exists():
        raise FileNotFoundError(
            f"Could not find file '{args.target_yaml_file}' in {dir}"
        )
    if not log_file.exists():
        raise FileNotFoundError(
            f"Could not find file '{args.target_log_file}' in {dir}"
        )

    return yaml_file, log_file


def parse_model_stats_and_metrics(yaml, log):
    stats_df = parse_stats(yaml)
    metrics_df = parse_metrics(log)
    return pd.concat(
        [stats_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1
    )


def exec_program(dir, args):
    print(f"Parsing files in {dir}")
    yaml_file, log_file = get_yaml_and_log_files(dir, args)
    df = parse_model_stats_and_metrics(yaml_file, log_file)

    if args.output_file.exists() and args.mode.startswith("a"):
        df_existing = pd.read_csv(args.output_file)
        df = pd.concat([df_existing, df], ignore_index=True)

    if len(args.sort_by) > 0:
        ascend = args.ascending
        if not ascend:
            ascend = True
        for col in args.sort_by:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(args.sort_by, ascending=ascend)
    df.to_csv(args.output_file, index=False, na_rep="None")
    print(f"Saved to {args.output_file}")


def collect_dirs(dirs, curr_dir, level):
    for dir in curr_dir.iterdir():
        if dir.is_file():
            continue
        if level == 0:
            dirs.append(dir)
        else:
            collect_dirs(dirs, dir, level - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, default=None)
    parser.add_argument("-g", "--target_log_file", type=str, required=True)
    parser.add_argument("-y", "--target_yaml_file", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-r", "--recursive", default=False, action="store_true")
    parser.add_argument("-m", "--mode", type=str, default="w")
    parser.add_argument("-l", "--levels", type=int, default=0)
    parser.add_argument(
        "-s",
        "--sort_by",
        type=lambda s: [s.strip() for s in s.split(",")],
        required=False,
    )
    parser.add_argument(
        "-a",
        "--ascending",
        type=lambda s: [bool(s.strip()) for s in s.split(",")],
        required=False,
    )

    args = parser.parse_args()

    if args.directory is None:
        args.directory = Path(os.getcwd()).resolve()
    else:
        args.directory = Path(args.directory).resolve()

    args.output_file = Path(args.output_file)
    args.target_yaml_file = Path(args.target_yaml_file)
    args.target_log_file = Path(args.target_log_file)

    if not args.recursive:
        print(f"Mode is {args.mode}")
        exec_program(args.directory, args)
    else:
        args.mode = "a+"

        level = args.levels
        dir_queue = []
        collect_dirs(dir_queue, args.directory, level)

        print(f"Total of {len(dir_queue)} directories")
        for dir in dir_queue:
            exec_program(dir, args)
