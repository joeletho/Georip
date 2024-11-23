import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import ArgumentError
from datetime import datetime
from pathlib import Path
from time import sleep

LOG_FILE = "yolo.out"
YOLO_FILE = "log.out"

device_stack = []


def get_timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def write_output(msg, file=None):
    output = f"[{get_timestamp()}] {msg}"
    if file:
        file.write(f"{output}\n")
    print(output)


def yolo_exec(dir, device, imgsz, yaml_path, args):
    """
    Executes the yolo command for a given directory and device, logs output to a file.
    Returns the process result, directory, and the device used.
    """
    os.chdir(dir)

    batch = args.batch
    if batch == -1.0:
        match (imgsz):
            case 320:
                batch = 16
            case 640:
                batch = 6
            case _:
                batch = 0.6

    with open(dir / YOLO_FILE, "w", buffering=1) as log_file:
        ret = subprocess.run(
            [
                "yolo",
                f"project={dir}",
                "mode=train",
                f"model={args.model}",
                f"imgsz={imgsz}",
                f"time={args.time}",
                f"patience={args.patience}",
                f"batch={batch}",
                f"freeze={args.freeze}",
                f"iou={args.iou}",
                f"save={args.save}",
                f"save_json={args.save_json}",
                f"save_conf={args.save_conf}",
                f"plots={args.plots}",
                f"exist_ok={args.exist_ok}",
                f"device={device}",
                f"data={yaml_path}",
            ],
            stdout=log_file,
            stderr=log_file,
        )
    return ret, dir, device


def main_execution_loop(cwd, dir_queue, device_stack, args):
    """
    Main loop that manages execution, devices, and processes.
    """
    with open(cwd / LOG_FILE, "w", buffering=1) as f:
        write_output("Starting", f)
        write_output(f"Devices: {', '.join(map(str, device_stack))}", f)
        write_output(f"Number of projects: {len(dir_queue)}", f)

        with ThreadPoolExecutor() as executor:
            futures = []
            while dir_queue or futures:
                while dir_queue and device_stack:
                    dir_path = dir_queue.pop()
                    device_id = device_stack.pop()
                    write_output(
                        f"Submitting project: {dir_path.name} on device {device_id}", f
                    )
                    ds_root = Path(args.dataset_root).resolve()
                    imgsz = args.imgsz
                    if isinstance(imgsz, str):
                        if imgsz != "parse":
                            raise ArgumentError(
                                f"Unknown argument for 'imgsz': {imgsz}"
                            )
                        if "imgsz" not in dir_path.name:
                            raise ValueError(
                                f"Cannot parse 'imgsz' from directory name: {dir_path.name}"
                            )
                        start = dir_path.name.find("imgsz")
                        end = dir_path.name.find("_", start)
                        imgsz = int(dir_path.name[start:end].split("=")[1])
                    futures.append(
                        executor.submit(
                            yolo_exec,
                            dir_path,
                            device_id,
                            imgsz,
                            ds_root / dir_path.name / args.data_path,
                            args,
                        )
                    )

                for future in as_completed(futures):
                    futures.remove(future)
                    try:
                        process, dir_path, device_id = future.result()
                        write_output(
                            f"Project {dir_path.name} finished on device {device_id} with return code {process.returncode}",
                            f,
                        )
                        device_stack.append(device_id)
                    except Exception as e:
                        write_output(f"Error with project: {e}", f)

                if not dir_queue and not futures:
                    write_output("All projects completed.", f)
                    break

                sleep(60)


def int_or_str(x):
    try:
        return int(x)
    except Exception:
        return str(x)

def int_or_float(n):
    if n.isdigit():
        return int(n)
    return float(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", type=str, required=True)
    parser.add_argument("-x", "--exclude", type=str, default=None, required=False)
    parser.add_argument("-c", "--data_path", type=str, required=True)
    parser.add_argument(
        "-D",
        "--device",
        type=lambda s: [int(s.strip()) for s in s.split(",")],
        required=True,
    )
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("-p", "--patience", type=int, default=100)
    parser.add_argument("-b", "--batch", type=lambda n: int_or_float(n), default=-1.0)
    parser.add_argument("-z", "--imgsz", type=lambda x: int_or_str(x), default="parse")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-f", "--freeze", type=int, default=None)
    parser.add_argument("-o", "--exist_ok", type=bool, default=False)
    parser.add_argument("-t", "--time", type=int, default=0)
    parser.add_argument("-i", "--iou", type=float, default=0.5)
    parser.add_argument("-S", "--save", type=bool, default=True)
    parser.add_argument("-J", "--save_json", type=bool, default=True)
    parser.add_argument("-C", "--save_conf", type=bool, default=True)
    parser.add_argument("-P", "--plots", type=bool, default=True)

    args = parser.parse_args()

    cwd = Path(os.getcwd()).resolve()
    device_stack.extend(args.device)

    dir_queue = []

    print(
        f"Starting... Output will be saved to '{LOG_FILE}'. Training output will be saved to '{YOLO_FILE}' in the project directory."
    )

    for dir in cwd.iterdir():
        if dir.is_file() or args.exclude and dir.name in args.exclude:
            continue
        dir_queue.append(dir)

    main_execution_loop(cwd, dir_queue, device_stack, args)
