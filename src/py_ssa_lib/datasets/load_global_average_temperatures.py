import pandas as pd
import os 
# https://www.kaggle.com/datasets/shishu1421/global-temperature/data
def load_global_average_temperatures_df(rawDS=False):

	dir_path = os.path.dirname(os.path.realpath(__file__))
	if rawDS:
		df_csv = os.path.join(dir_path,"GlobalLandTemperaturesByMajorCity.csv")
		return pd.read_csv(filepath_or_buffer=df_csv , sep=",")
	else:
		df_csv = os.path.join(dir_path,"GlobalLandTemperaturesByMajorCity.csv")
		glob_temp_df = pd.read_csv(filepath_or_buffer=df_csv, sep=",")
		glob_temp_df = glob_temp_df.dropna(axis=0)
		glob_temp_df.dt = pd.to_datetime(glob_temp_df.dt)
		cols_list = glob_temp_df["dt"]
		glob_temp_df = glob_temp_df.drop(columns=["dt"]).T
		glob_temp_df.columns = cols_list.values
			
		return glob_temp_df

				
