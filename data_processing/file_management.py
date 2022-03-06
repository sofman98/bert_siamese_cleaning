import pandas as pd

# LOAD DATASET

# parquet
# for loading the dataset
def load_dataset_parquet(path='datasets/cs1_us_outlets.parquet.gzip'):
  return pd.read_parquet(path)

# for saving parquet files
def save_parquet(dataframe, path):
  # dataframe.to_parquet(path)
  # I prefer csv
  dataframe.to_csv(path, index=False)

# csv
# for loading the dataset
def load_dataset_csv(path):
  return pd.read_csv(path)

# for saving csv files
def save_csv(dataframe, path):
  dataframe.to_csv(path, index=False)
