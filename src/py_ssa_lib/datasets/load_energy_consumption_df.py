import pandas as pd
import os 

def load_energy_consumption_df(rawDS=False):

	dir_path = os.path.dirname(os.path.realpath(__file__))
	if rawDS:
		df_csv = os.path.join(dir_path,"total-primary-energy-consumption-csv-version-4.csv")
		return pd.read_csv(filepath_or_buffer=df_csv , sep=",")
	else:
		df_csv = os.path.join(dir_path,"energy_consumption_df.csv")
		return pd.read_csv(filepath_or_buffer=df_csv, sep=",")
	
		
	