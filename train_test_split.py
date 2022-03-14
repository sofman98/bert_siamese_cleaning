<<<<<<< HEAD
from data_processing.file_management import load_dataset_csv, load_dataset_parquet, save_csv
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset

features = ['name']
train_data_save_path = 'datasets/train.csv'
test_data_save_path = 'datasets/test.csv'
train_features_save_path = 'datasets/entire_feature_similarity_train.csv'
test_features_save_path = 'datasets/entire_feature_similarity_test.csv'

test_frac = 0.2
if __name__ == "__main__":
    # we load the dataset
    dataset = load_dataset_csv('datasets/cs1_us_outlets.csv')
    # or load the .parquet.gzip file
    # dataset = load_dataset_parquet('datasets/cs1_us_outlets.parquet.gzip')

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
        kind='permutations',
        NUM_NEG=None,
        features=features,
        max_neg=True,
        save_to=train_features_save_path
    )

    _ = generate_feature_similarity_dataset(
        test,
        kind='combinations',
        NUM_NEG=None,
        features=features,
        max_neg=True,
        save_to=test_features_save_path
    )
=======
from data_processing.file_management import load_dataset_csv, load_dataset_parquet, save_csv
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset

features = ['lat', 'lon']
train_data_save_path = 'datasets/train.csv'
test_data_save_path = 'datasets/test.csv'
train_features_save_path = 'datasets/entire_feature_similarity_train.csv'
test_features_save_path = 'datasets/entire_feature_similarity_test.csv'

test_frac = 0.2
if __name__ == "__main__":
    # we load the dataset
    dataset = load_dataset_csv('datasets/cs1_us_outlets.csv')
    # or load the .parquet.gzip file
    # dataset = load_dataset_parquet('datasets/cs1_us_outlets.parquet.gzip')

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
    # all_neg_combinations=True, which means it includes all possible combinations 
    # in a same persistent_cluster for negative values (non-similarity)

    _ = generate_feature_similarity_dataset(
        train,
        features=features,
        NUM_NEG=None,
        all_neg_combinations=True,
        save_to=train_features_save_path
    )

    _ = generate_feature_similarity_dataset(
        test,
        features=features,
        NUM_NEG=None,
        all_neg_combinations=True,
        save_to=test_features_save_path
    )
>>>>>>> main
