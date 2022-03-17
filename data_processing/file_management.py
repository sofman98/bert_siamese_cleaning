import pandas as pd
import os, errno

# LOAD DATASET

# parquet
# for loading the dataset
def load_parquet(path):
  return pd.read_parquet(path)

def save_parquet(dataframe, path):
  dataframe.to_parquet(path, index=False)

# csv
# for loading the dataset
def load_csv(path):
  return pd.read_csv(path)

# for saving csv files
def save_csv(dataframe, path):
  dataframe.to_csv(path, index=False)

# for creating folders

def create_folder(path):
	if not os.path.exists(os.path.dirname(path)):
		if not os.path.exists(os.path.dirname(path)):
			try:
				os.makedirs(os.path.dirname(path))
			except OSError as exc:  # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise