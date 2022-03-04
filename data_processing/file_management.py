import pandas as pd

# TYPE PARQUET

# for loading the dataset
def load_dataset(path='datasets/cs1_us_outlets.parquet.gzip'):
  return pd.read_parquet(path)

# for saving parquet files
def save_parquet(dataframe, path):
  # dataframe.to_parquet(path)
  # I prefer csv
  dataframe.to_csv(path)



# TYPE CSV
def load_dataset_csv(path='datasets/cs1_us_outlets.parquet.gzip'):
  return pd.read_csv(path)

# for saving parquet files
def save_csv(dataframe, path):
  dataframe.to_csv(path)