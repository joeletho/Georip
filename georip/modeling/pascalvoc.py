import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm, trange

from georip.modeling.utils import XMLTree
from georip.utils import NUM_CPU


def pascal_xml_annotation_to_csv(xmlpath, imgdir, destpath):
    columns = [
        "image_filename",
        "image_width",
        "image_height",
        "image_path",
        "class_name",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    ]

    rows = []

    for i in XMLTree(xmlpath).root().findall("."):
        filename = i.find("filename")
        filename = "null" if filename is None else filename.text
        image_path = "null"

        if filename is not None:
            origin = Path(imgdir, filename).resolve()
            orig_stem = origin.stem
            for file in os.listdir(imgdir):
                filepath = Path(imgdir, file).resolve()
                if filepath.is_file() and filepath.stem == orig_stem:
                    image_path = filepath
                    break

        if image_path == "null":
            raise Exception(f"Image path is missing '{filename}'")

        class_name = "null"
        bboxes = []

        width = i.find("./size/width").text
        height = i.find("./size/height").text

        for obj in i.findall("object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            bboxes.append(
                [
                    bbox.find("xmin").text,
                    bbox.find("ymin").text,
                    bbox.find("xmax").text,
                    bbox.find("ymax").text,
                ]
            )

        for bbox in bboxes:
            rows.append(
                {
                    "image_filename": filename,
                    "image_width": width,
                    "image_height": height,
                    "image_path": image_path,
                    "class_name": class_name,
                    "bbox_xmin": bbox[0],
                    "bbox_ymin": bbox[1],
                    "bbox_xmax": bbox[2],
                    "bbox_ymax": bbox[3],
                }
            )
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(destpath)


def pascal_xml_annotation_to_dataframe(xmlpath):
    columns = [
        "filename",
        "width",
        "height",
        "name",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    rows = []

    for i in XMLTree(xmlpath).root().findall("."):
        filename = i.find("filename")
        filename = "null" if filename is None else filename.text
        names = []
        bboxes = []

        width = i.find("./size/width").text
        height = i.find("./size/height").text

        for obj in i.findall("object"):
            names.append(obj.find("name").text)
            bbox = obj.find("bndbox")
            bboxes.append(
                [
                    bbox.find("xmin").text,
                    bbox.find("ymin").text,
                    bbox.find("xmax").text,
                    bbox.find("ymax").text,
                ]
            )

        for i, bbox in enumerate(bboxes):
            rows.append(
                {
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "name": names[i],
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def pascal_process_xml_files_to_dataframe(xmldir, *, parallel=True):
    files = []
    for file in os.listdir(xmldir):
        filepath = Path(xmldir, file).resolve()
        if filepath.is_file() and filepath.suffix == ".xml":
            files.append(filepath)

    df = pd.DataFrame()
    nfiles = len(files)
    batch = 8
    chunksize = nfiles // batch

    def __process_chunk__(files, n):
        df = pd.DataFrame()
        for file in tqdm(
            files,
            desc=(
                "Processing XML files to XML DataFrame"
                if n is None
                else f"{f'Batch {n}' : <15}"
            ),
            leave=False,
        ):
            df = pd.concat([df, pascal_xml_annotation_to_dataframe(file)])
        return df

    if not parallel:
        return __process_chunk__(files, None)

    pbar = tqdm(
        total=batch,
        desc="Processing XML files to XML DataFrame",
        leave=False,
    )
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        count = 1
        futures = []
        for i in range(0, nfiles, chunksize):
            start = i
            end = i + chunksize
            futures.append(executor.submit(__process_chunk__, files[start:end], count))
            count += 1

        for future in as_completed(futures):
            df = pd.concat([df, future.result()])
            pbar.update()
    pbar.close()
    return df


def pascal_xml_to_dataframe(xmldir, imgdir, *, parallel=True):
    pbar = trange(1, position=0, desc="Progress")

    df = pascal_process_xml_files_to_dataframe(xmldir, parallel=parallel)
    pbar.update()
    pbar.close()

    return df


def pascal_process_xml_to_csv_files(csvdir, imgdir, destdir):
    files = []
    for file in os.listdir(csvdir):
        filepath = Path(csvdir, file).resolve()
        if filepath.is_file() and filepath.suffix == ".csv":
            files.append(filepath)

    nfiles = len(files)
    nthreads = NUM_CPU
    chunksize = nfiles // nthreads

    def __process_chunk__(files, _imgdir, _destdir):
        for file in files:
            pascal_xml_annotation_to_csv(file, _imgdir, _destdir)

    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        i = 0
        while i < nfiles:
            start = i
            end = i + chunksize
            executor.submit(__process_chunk__, files[start:end], imgdir, destdir)
            i = end
