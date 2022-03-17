from data_processing.file_management import load_dataset_csv, load_dataset_parquet, save_csv
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
import numpy as np

# important variables
test_frac = 0.2
feature = 'name'

# train data config
train_NUM_NEG = 5 # number of negative instances per outlet - Dataset is balanced if NUM_NEG==1
train_max_neg = True # if True then NUM_NEG is useless, uses all negative instance, if False then considers NUM_NEG

# loading path
path_to_dataset = 'datasets/cs1_us_outlets.csv'
path_to_encodings = 'datasets/encodings.npy'

# saving path
train_data_save_path = 'datasets/train.csv'
test_data_save_path = 'datasets/test.csv'
train_features_save_path = 'datasets/entire_feature_similarity_train.npy'
test_features_save_path = 'datasets/entire_feature_similarity_test.npy'


if __name__ == "__main__":
    # we load the dataset and the generated encodings
    # or load the .parquet.gzip file
    # dataset = load_dataset_parquet('datasets/cs1_us_outlets.parquet.gzip')
    dataset = load_dataset_csv(path_to_dataset)
    encodings = np.load(path_to_encodings)

    # we add the encodings data to the dataset
    dataset[f'{feature}_encoding'] = list(encodings)

    # we split it
    test = dataset.sample(frac=test_frac, random_state=0)
    train = dataset.drop(test.index)

    # we reset the index not to be out of bounds
    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)

    # saving the test and train sets
    save_csv(train, train_data_save_path)
    save_csv(test, test_data_save_path)

    # next we generate the feature similarity dataset (feature_set1, feature_set2, similarity)
    # max_neg=True, which means it includes all possible combinations 
    # in a same persistent_cluster for negative values (non-similarity)
    # we select all possible permutations for the training
    # and all unique combinations for the test

    _ = generate_feature_similarity_dataset(
        train,
        kind='combinations',
        NUM_NEG=train_NUM_NEG,
        features=[f'{feature}_encoding'],
        max_neg=train_max_neg,
        save_to=train_features_save_path
    )

    _ = generate_feature_similarity_dataset(
        test,
        kind='combinations',
        NUM_NEG=None,
        features=[f'{feature}_encoding'],
        max_neg=True,
        save_to=test_features_save_path
    )
