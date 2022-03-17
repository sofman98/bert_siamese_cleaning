import numpy as np
from data_processing.data_processing import preprocess_dataset
from data_processing.data_processing import filter_features
import itertools as it

def generate_positive_data(
    dataset,
  ):
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
      # we get all combinations
      # example: [5, 6, 7] => [5, 6], [5, 7], [6, 7], [6, 5]..etc
      if len(indexes) > 1:
        indexes_permutation = np.array(list(it.permutations(indexes, 2)))
        # we add a (1) to signify similarity
        ones = np.ones(shape=(indexes_permutation.shape[0], 1), dtype=int)
        pos_instances = np.append(indexes_permutation, ones, axis=1)
        for instance in pos_instances:
          positive_data.append(instance)
  
  return positive_data

def generate_negative_data(
    dataset,
    NUM_NEG,
  ):
  """
  adding the negatives.
  we randomly select a number of non-similar places *in the same persistent_cluster* to act as a negative [id1, id2, 0].
  we specifically chose places in the same cluster as they would be more difficult to predict when training on latitude and longitude.
  """
  assert NUM_NEG >= 0
  
  negative_data = []
  #for every outlet
  for index, row in dataset.iterrows():
    out1_cluster = row.persistent_cluster
    out1_id_dashmote = row.id_dashmote
    # look for outlets in the same persistent_cluster as out1
    ## but that have a different id_dashmote
    same_cluster_different_outlets = dataset[
        (dataset.persistent_cluster == out1_cluster)
        & (dataset.id_dashmote != out1_id_dashmote)
    ]

    # we only sample NUM_NEG number of outlets
    ## sometimes NUM_NEG > len(same_cluster_different_outlets)
    ## so we take the min of the two
    num_samples = min(NUM_NEG, len(same_cluster_different_outlets))
    same_cluster_different_outlets = same_cluster_different_outlets.sample(n=num_samples, random_state=0).index.to_list()

    for out2_index in same_cluster_different_outlets:
      negative_data.append([index, out2_index, 0])
      
  return negative_data


def generate_pair_similarity_num_neg(
    dataset, 
    NUM_NEG,
  ):
  """
  We generate the positive and negative data and combine them.
  we define the parameter NUM_NEG or number of negatives instances for every outlet
  """
  assert NUM_NEG >= 0

  print("Processing dataset...")
  dataset = preprocess_dataset(dataset)

  print('Generating positive instances..')
  positive_data = generate_positive_data(
    dataset,
  )
  
  print(f'Generating negative instances with {NUM_NEG} negative instances for every outlet..')
  
  negative_data = generate_negative_data(
    dataset,
    NUM_NEG,
  )

  # combining the positive and negative data to form a pair similarity dataset
  pair_similarity_dataset = np.concatenate([positive_data, negative_data], axis = 0)

  return pair_similarity_dataset


def generate_entire_pair_similarity(
    dataset,
    kind,
  ):
  """
  Selects entire pair_similarity without considering NUM_NEG.
  """
  # some verfifications
  assert kind in ['combinations', 'permutations']

  # we select all unique clusters
  unique_clusters = dataset.persistent_cluster.unique()

  pair_similairty_dataset = []
  # for each persistent_cluster
  for cluster in unique_clusters:
    # get all indexes
    cluster_outlets = dataset[dataset.persistent_cluster == cluster].index
    # find all combinations or permutations
    if kind=='combinations':
      indexes_pairs = list(it.combinations(cluster_outlets, 2))
    else:
      indexes_pairs = list(it.permutations(cluster_outlets, 2))
    pair_similairty_dataset += indexes_pairs

  pair_similairty_dataset = np.array(pair_similairty_dataset)
  # we select the indexes
  outlets_1 = pair_similairty_dataset[:,0]
  outlets_2 = pair_similairty_dataset[:,1]
  # we select their id_dashmote
  outlets_1_id_dashmote = dataset.iloc[outlets_1].id_dashmote.to_numpy()
  outlets_2_id_dashmote = dataset.iloc[outlets_2].id_dashmote.to_numpy()
  # we compare the id_dashmote, 1 if similar, 0 if different
  similarity = np.array(outlets_1_id_dashmote == outlets_2_id_dashmote, dtype=int).reshape(-1,1)
  pair_similairty_dataset = np.append(pair_similairty_dataset, similarity, axis=1)

  return pair_similairty_dataset


def generate_feature_similarity_dataset(
    dataset,
    features,
    NUM_NEG,
    kind,
    max_neg=True,
    save_to=''    
  ):
  """
  This function transforms the data into a feature similarity dataset.
  The generated data are in the form (feature_set1, feature_set2, similarity).
  With "similarity" being 0 or 1.
  NUM_NEG represents the number of negative instances for every outlet
  """
  # some verfifications
  assert kind in ['combinations', 'permutations']
  assert NUM_NEG >= 0
  assert max_neg in [True, False]

  # we generate a pair similarity dataset (outlet1, outlet2, similarity)
  if max_neg:
    pair_similarity_dataset = generate_entire_pair_similarity(dataset, kind)
  else:
    pair_similarity_dataset = generate_pair_similarity_num_neg(dataset, NUM_NEG) # kind is permutations by default
    
  # we select the desired features
  # we squeeze it to get rid of the undesired dimension in axis 1
  filtered_dataset = np.array(filter_features(dataset, features).values.tolist()).squeeze(axis=1)


  # we get the feature values for all pairs
  outlet1 = filtered_dataset[pair_similarity_dataset[:, 0]]
  outlet2 = filtered_dataset[pair_similarity_dataset[:, 1]]
  similarity = pair_similarity_dataset[:, -1].reshape((-1, 1))

  feature_similarity_dataset = np.concatenate([outlet1, outlet2, similarity], axis=1)

  #saving the new generate dataset, if given a path
  if save_to:
    print(f'Saving the feature similarity dataset to {save_to}..')
    np.save(save_to, feature_similarity_dataset)

  return feature_similarity_dataset
  