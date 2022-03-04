from sklearn import preprocessing

def label_encode(dataset, columns):
  """
  This is for label encoding columns,
  make them in the (0 to n-1 classes) range
  """
  le = preprocessing.LabelEncoder()
  #for every column we want to encode
  for col in columns:
    dataset[col] = le.fit_transform(dataset[col].values)
  
  return dataset

def normalize_lat_lon(dataset):
  """
  normalizing latitute and longitude
  we use the maximum real-world values so that it's compatible with new locations
  """
  (MIN_LAT, MAX_LAT) = (-90, 90)
  (MIN_LON, MAX_LON) = (-180, 180)

  dataset.lat = (dataset.lat - MIN_LAT) / (MAX_LAT - MIN_LAT)
  dataset.lon = (dataset.lon - MIN_LON) / (MAX_LON - MIN_LON)

  return dataset

def process_dataset(
      dataset,
      encode_columns=['id_dashmote', 'persistent_cluster'],
      normalize_latitude_and_longitude=True
    ):
  """
  We combine the processing functions into one.
  """
  # we label encode the columns we want to encode
  dataset = label_encode(dataset, encode_columns)

  # we normalize the latitude and longitude
  if normalize_latitude_and_longitude:
    dataset = normalize_lat_lon(dataset)

  # we finally return the processed dataset
  return dataset