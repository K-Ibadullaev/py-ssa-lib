import pandas as pd
import os 
# https://www.kaggle.com/datasets/shishu1421/global-temperature/data
from importlib.resources import files
import pandas as pd

def load_global_average_temperatures_df(rawDS=False):

    csv_path = (
        files("py_ssa_lib.datasets")
        / "GlobalLandTemperaturesByMajorCity.csv"
    )

    glob_temp_df = pd.read_csv(csv_path)

    if rawDS:
        return glob_temp_df

    glob_temp_df = glob_temp_df.dropna(axis=0)
    glob_temp_df.dt = pd.to_datetime(glob_temp_df.dt)

    cols_list = glob_temp_df["dt"]

    glob_temp_df = glob_temp_df.drop(
        columns=["dt"]
    ).T

    glob_temp_df.columns = cols_list.values

    return glob_temp_df

				
