import warnings
from typing import Union

import geopandas as gpd
import pandas as pd
from osgeo import gdal

DataFrameLike = Union[pd.DataFrame, gpd.GeoDataFrame]


warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)
gdal.UseExceptions()
