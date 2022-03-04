import numpy as np
import pandas as pd
import random
from data_processing.data_processing import process_dataset
from data_processing.file_management import save_parquet

def filter_features(
      dataset,
      features=[]
    ):
  """we filter the dataset according desired features"""
  # we remove the incorrect feature names
  for index, feature in enumerate(features):
    if feature not in dataset.columns:
        print(f'Error: no feature named {feature}.')
        features.pop(index)

  # we provide default values
  ## if features is empty, keep the original features
  if features == []:
    features = list(dataset.columns)
  
  return dataset[features]

def generate_positive_data(dataset):
  """
  changing dataset into outlet1, outlet2, 1.
  with 1 representing the similarity between outlet1 and outlet2
  """
  # we use a map for better complexity: O(n)
  # we save them as {id_dashmote: [id1, id2...id3]} for the similar ones
  similarity_map = {}

  for index, row in dataset.iterrows():
      id_dashmote = int(row.id_dashmote)
      if id_dashmote in similarity_map:
          similarity_map[id_dashmote].append(index)
      else:
          similarity_map[id_dashmote] = [index]

  # creating the similarity dataset which would be like (id1, id2, 1) 
  positive_data = []
  for indexes in similarity_map.values():
      # we go twice through the list of similar places to save all the similar ones
      # example: [5, 6, 7] => [5, 6], [5, 7], [6, 5], [6, 7] ...etc
      for i in indexes:
          for j in indexes:
              if i != j:
                  positive_data.append([i, j, 1])

  return np.stack(positive_data)

def generate_negative_data(
      dataset, 
      NUM_NEG,
      number_positive_instances
    ):
    """
    adding the negatives.
    we randomly select a number of non-similar places *in the same persistent_cluster* to act as a negative [id1, id2, 0].
    we specifically chose places in the same cluster as they would be more difficult to predict when training on latitude and longitude.
    we define the parameter NUM_NEG or number of negatives instances per one positive instance
    for example, if NUM_NEG=1, then the dataset is balanced, if NUM_NEG=2, then there are twice more negative instances than positive ones
    """

    negative_data = []
    # we stop when we have enough negative instances
    while len(negative_data) != (number_positive_instances * NUM_NEG):   # len() is O(1) so it's ok to call it every time
      # randomly select one outlet from the dataset
      out1 = random.randint(0, dataset.shape[0]-1)
      out1_id_dashmote = dataset.iloc[out1].id_dashmote

      # look for outlets in the same block as out1
      ## but that have a different id_dashmote
      out1_persistent_cluster = dataset.iloc[out1].persistent_cluster

      same_cluster_different_outlets = dataset[
          (dataset.persistent_cluster == out1_persistent_cluster)
          & (dataset.id_dashmote != out1_id_dashmote)
      ]
      # we randomly sample from these outlets
      out2 = int(same_cluster_different_outlets.sample().id_dashmote)
      negative_dataset.append([out1_id_dashmote, out2_id_dashmote, 0])
        
    return np.stack(negative_data)

def generate_pair_similarity_dataset(
      dataset, 
      NUM_NEG=1,
      save_to=''
    ):
  """
  We generate the positive and negative data and combine them.
  NUM_NEG==1 by default, the dataset is balanced
  """
  print("Processing dataset...")
  dataset = process_dataset(dataset)

  print('Generating pair similarity dataset..')
  print('Generating positive instances..')
  positive_data = generate_positive_data(dataset)

  print(f'Generating negative instances with {NUM_NEG} negative instance per positive one..')
  negative_data = generate_negative_data(
    dataset, 
    NUM_NEG,
    len(positive_data)
  )

  # combining the positive and negative data to form a pair similarity dataset
  pair_similarity_dataset = np.concatenate([positive_data, negative_data], axis = 0)
  pair_similarity_dataset = np.stack(pair_similarity_dataset)

  # converting back to DataFrame
  pair_similarity_df = pd.DataFrame(data=pair_similarity_dataset, columns=['outlet1', 'outlet2', 'similarity'])

  #saving the new generate dataset, if given a path
  if save_to:
    print(f'Saving the pair similarity dataset to {save_to}..')
    save_parquet(pair_similarity_df, save_to)

  return pair_similarity_df

def generate_feature_similarity_dataset(
      dataset,
      features,
      NUM_NEG=1,
      save_to=''    
    ):
  """
  This function transforms the data into a feature similarity dataset.
  The generated data are in the form (feature_set1, feature_set2, similarity).
  With "similarity" being 0 or 1.
  NUM_NEG==1 by default, the dataset is balanced.
  """
  # we generate a pair similarity dataset (outlet1, outlet2, similarity)
  pair_similarity_df = generate_pair_similarity_dataset(
    dataset,
    NUM_NEG,
  )

  # we select the desired features
  filtered_dataset = filter_features(dataset, features)

  # we select the outlet features for each of the outlets (feature_set1, feature_set2, similarity)
  feature_similarity_dataset = []
  for index, row in pair_similarity_df.iterrows():
      # we retrieve the data from the original dataset
      out1_data = filtered_dataset.iloc[row.outlet1]
      out2_data = filtered_dataset.iloc[row.outlet2]
      # selecting the features and converting to list
      out1_data = list(out1_data)
      out2_data = list(out2_data)
      # combining the two and appending them to the stack
      data_instance = out1_data + out2_data + [int(row.similarity)]
      feature_similarity_dataset.append(data_instance)

  feature_similarity_dataset = np.stack(feature_similarity_dataset)

  #transforming into dataframe
  columns = [ col + f'{i}' for col in list(filtered_dataset.columns) for i in [1, 2]] + ['similarity']
  feature_similarity_df = pd.DataFrame(data=feature_similarity_dataset, columns=columns)

  #saving the new generate dataset, if given a path
  if save_to:
    print(f'Saving the feature similarity dataset to {save_to}..')
    save_parquet(feature_similarity_df, save_to)

  return feature_similarity_df
