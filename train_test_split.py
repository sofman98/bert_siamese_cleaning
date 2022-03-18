from data_processing.file_management import load_parquet, save_parquet
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
import numpy as np

# important variables
test_frac = 0.2
feature = 'name'

# train data config
train_NUM_NEG = 5 # number of negative instances per outlet - Dataset is balanced if NUM_NEG==1
train_max_neg = True # if True then NUM_NEG is useless, uses all negative instance, if False then considers NUM_NEG
train_kind = 'combinations' # 'combinations' or 'permutations'

# loading path
path_to_dataset = 'datasets/cs1_us_outlets.parquet.gzip'
path_to_embeddings = 'datasets/embeddings.npy'

# saving path
train_data_save_path = 'datasets/train.parquet.gzip'
test_data_save_path = 'datasets/test.parquet.gzip'
train_features_save_path = 'datasets/feature_similarity_train.npy'
test_features_save_path = 'datasets/feature_similarity_test.npy'


if __name__ == "__main__":
  # we load the dataset and the generated embeddings
  dataset = load_parquet(path_to_dataset)
  embeddings = np.load(path_to_embeddings)

  # we split it
  test = dataset.sample(frac=test_frac, random_state=0)
  train = dataset.drop(test.index)

  # saving the test and train sets
  save_parquet(train, train_data_save_path)
  save_parquet(test, test_data_save_path)

  # we add the embeddings
  test[f'{feature}_embedding'] = list(embeddings[test.index])
  train[f'{feature}_embedding'] = list(embeddings[train.index])

  # we reset the index not to be out of bounds
  test.reset_index(drop=True, inplace=True)
  train.reset_index(drop=True, inplace=True)

  # next we generate the feature similarity dataset (feature_set1, feature_set2, similarity)
  # max_neg=True, which means it includes all possible combinations 
  # in a same persistent_cluster for negative values (non-similarity)
  # we select all possible permutations or combinations for the training
  # and all unique combinations for the test

  _ = generate_feature_similarity_dataset(
      train,
      kind=train_kind,
      NUM_NEG=train_NUM_NEG,
      features=[f'{feature}_embedding'],
      max_neg=train_max_neg,
      save_to=train_features_save_path
  )

  _ = generate_feature_similarity_dataset(
      test,
      kind='combinations',
      NUM_NEG=None,
      features=[f'{feature}_embedding'],
      max_neg=True,
      save_to=test_features_save_path
  )
