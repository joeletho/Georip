from rasterio import DatasetReader


def get_rows_cols_min_max_bounds(
    dataset: DatasetReader,
) -> tuple[tuple[int, int], tuple[int, int]]:
    bounds = dataset.bounds
    rmin, cmin = dataset.index(bounds.left, bounds.top)
    rmax, cmax = dataset.index(bounds.right, bounds.bottom)
    rmin, rmax = min(rmin, rmax), max(rmin, rmax)
    cmin, cmax = min(cmin, cmax), max(cmin, cmax)
    return (rmin, rmax), (cmin, cmax)
