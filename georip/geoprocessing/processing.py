from typing import Union

import geopandas as gpd
from tqdm.auto import trange

from georip.geometry.polygons import flatten_polygons
from georip.geoprocessing.mapping import map_metadata
from georip.geoprocessing.utils import gdf_matches_image_crs, gdf_set_crs_to_image
from georip.io import collect_files_with_suffix
from georip.io.geoprocessing import load_shapefile
from georip.utils import StrPathLike


def preprocess_ndvi_shapefile(
    source_path: StrPathLike,
    *,
    years: None | tuple[int, int],
    region_col: str | list[str],
    start_year_col: str,
    end_year_col: str,
    images_dir: StrPathLike,
    preserve_fields: Union[
        list[Union[str, dict[str, str]]],
        dict[str, str],
        None,
    ] = None,
) -> gpd.GeoDataFrame:
    """
    Preprocesses a shapefile by flattening polygons, removing duplicates, and mapping metadata.

    Parameters:
        source_path (StrPathLike): The path to the input shapefile to be processed.
        years (tuple[int, int] | None): A tuple indicating the start and end year for filtering rows. If None, no filtering is applied.
        region_col (str): The column name representing the region in the shapefile.
        start_year_col (str): The column name representing the start year in the shapefile.
        end_year_col (str): The column name representing the end year in the shapefile.
        images_dir (StrPathLike): The directory containing image data for metadata mapping.
        filter_geometry (bool): Whether to filter out empty geometries.
        preserve_fields (list | dict | None): Fields to preserve, either as a list of strings/dictionaries
                                              or a dictionary mapping input to output names.

    Returns:
        gpd.GeoDataFrame: A processed GeoDataFrame with flattened polygons, unique geometries,
                          and mapped metadata.
    """
    gdf = load_shapefile(source_path)

    base_images = collect_files_with_suffix([".tif", ".tiff"], images_dir, recurse=True)
    if not len(base_images):
        raise FileNotFoundError(f"No '.tif' or '.tiff' images found in {images_dir}")

    if not gdf_matches_image_crs(gdf, base_images):
        gdf_set_crs_to_image(gdf, base_images[0])
        if not gdf_matches_image_crs(gdf, base_images):
            raise ValueError(
                "Source shapefile and source images have differing Coordinate Reference Systems"
            )

    # If years are provided, filter the rows based on the start and end years.
    if years is not None:
        start_year, end_year = years
        start_year = int(start_year)
        end_year = int(end_year)
        gdf = gdf[(gdf[start_year_col] >= start_year) & (gdf[end_year_col] <= end_year)]

    pbar = trange(2, desc="Flattening polygons and mapping metadata", leave=False)

    gdf = flatten_polygons(gdf)
    pbar.update()

    gdf = map_metadata(
        gdf,
        images_dir=images_dir,
        region_column=region_col,
        start_year_column=start_year_col,
        end_year_column=end_year_col,
        preserve_fields=preserve_fields,
    )
    pbar.update()
    pbar.close()

    if not len(gdf):
        raise ValueError("Shapefile does not contain valid metadata")

    return gdf
