import numpy as np
import pandas as pd
import random
from data_processing.data_processing import process_dataset
from data_processing.file_management import save_csv
import itertools as it

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

  # creating the similarity dataset which would be of the form (id1, id2, 1) 
  positive_data = []
  for indexes in similarity_map.values():
      # we go twice through the list of similar places to save all the similar ones
      # example: [5, 6, 7] => [5, 6], [5, 7], [6, 5], [6, 7] ...etc
      for i in indexes:
          for j in indexes:
              if i != j:
                  positive_data.append([i, j, 1])
  # creating the similarity dataset which would be of the form (id1, id2, 1) 
  # positive_data = []
  # for indexes in similarity_map.values():
  #     # we get unique combinations
  #     # example: [5, 6, 7] => [5, 6], [5, 7], [6, 7] ...etc
  #     if len(indexes) > 1:
  #       indexes_combinatons = np.array(list(it.combinations(indexes, 2)))
  #       # we add a (1) to signify similarity
  #       ones = np.ones(shape=(indexes_combinatons.shape[0], 1), dtype=int)
  #       pos_instances = np.append(indexes_combinatons, ones, axis=1)
  #       for instance in pos_instances:
  #         positive_data.append(instance)
  
  positive_data = np.array(positive_data)
  # positive_data.reshape((-1, 3))  
  return positive_data

def generate_negative_data(
    dataset,
    NUM_NEG
  ):
  """
  adding the negatives.
  we randomly select a number of non-similar places *in the same persistent_cluster* to act as a negative [id1, id2, 0].
  we specifically chose places in the same cluster as they would be more difficult to predict when training on latitude and longitude.
  we define the parameter NUM_NEG or number of negatives instances for every outlet
  for example, if NUM_NEG=1, then the dataset is balanced, if NUM_NEG=2, then there are twice more negative instances than positive ones
  if NUM_NEG < 0, we select all possible combinations from same persistent_cluster (best)
  """

  negative_data = []
  #for every outlet
  for index, row in dataset.iterrows():
    out1_cluster = row.persistent_cluster
    out1_id_dashmote = row.id_dashmote
    # look for outlets in the same block as out1
    ## but that have a different id_dashmote
    same_cluster_different_outlets = dataset[
        (dataset.persistent_cluster == out1_cluster)
        & (dataset.id_dashmote != out1_id_dashmote)
    ]

    # we use all data if NUM_NEG < 0
    if NUM_NEG < 0:
      # we take all the data
      same_cluster_different_outlets = same_cluster_different_outlets.index.to_list()
    else:
      # we only sample NUM_NEG number of instances
      ## sometimes NUM_NEG > len(same_cluster_different_outlets)
      ## so we take the min of the two
      sample_num = min(NUM_NEG, len(same_cluster_different_outlets))
      same_cluster_different_outlets = same_cluster_different_outlets.sample(n=sample_num).index.to_list()

    for out2_index in same_cluster_different_outlets:
      negative_data.append([index, out2_index, 0])
      
  return negative_data


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

  
  if NUM_NEG >= 0:
    print(f'Generating negative instances with {NUM_NEG} negative instances for every outlet..')
  else:
    print('Generating all possible negative instace combinations of every persistent_cluster')
  
  if NUM_NEG==0:
    # no need to generate negative data
    pair_similarity_dataset = positive_data
  else:
    negative_data = generate_negative_data(
      dataset,
      NUM_NEG
    )

    # combining the positive and negative data to form a pair similarity dataset
    pair_similarity_dataset = np.concatenate([positive_data, negative_data], axis = 0)

  # converting back to DataFrame
  pair_similarity_df = pd.DataFrame(data=pair_similarity_dataset, columns=['outlet1', 'outlet2', 'similarity'])

  #saving the new generate dataset, if given a path
  if save_to:
    print(f'Saving the pair similarity dataset to {save_to}..')
    save_csv(pair_similarity_df, save_to)

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
  NUM_NEG represents the number of negative instances for every outlet
  """
  # we generate a pair similarity dataset (outlet1, outlet2, similarity)
  pair_similarity_df = generate_pair_similarity_dataset(
    dataset,
    NUM_NEG,
  )

  # we select the desired features
  filtered_dataset = filter_features(dataset, features)
  # numpy for faster training
  filtered_dataset_np = filtered_dataset.to_numpy()

  # we select the outlet features for each of the outlets (feature_set1, feature_set2, similarity)
  feature_similarity_dataset = []
  for index, row in pair_similarity_df.iterrows():
      # we retrieve the data from the original dataset
      out1_data = filtered_dataset_np[row.outlet1]
      out2_data = filtered_dataset_np[row.outlet2]

      # combining the two and appending them to the stack
      data_instance = list(out1_data) + list(out2_data) + [int(row.similarity)]
      feature_similarity_dataset.append(data_instance)

  feature_similarity_dataset = np.stack(feature_similarity_dataset)

  #transforming into dataframe
  columns = [ col + f'{i}' for col in list(filtered_dataset.columns) for i in [1, 2]] + ['similarity']
  feature_similarity_df = pd.DataFrame(data=feature_similarity_dataset, columns=columns)

  #saving the new generate dataset, if given a path
  if save_to:
    print(f'Saving the feature similarity dataset to {save_to}..')
    save_csv(feature_similarity_df, save_to)

  return feature_similarity_df
  