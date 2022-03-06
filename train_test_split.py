from data_processing.file_management import load_dataset_csv, save_csv
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

    # we split it
    test = dataset.sample(frac=test_frac, random_state=0)
    train = dataset.drop(test.index)

    # saving the test and train sets
    save_csv(train, train_data_save_path)
    save_csv(test, test_data_save_path)

    # next we generate the feature similarity dataset (feature_set1, feature_set2, similarity)
    # NUM_NEG==-1 which means it includes all possible combinations 
    # in a same persistent_cluster for negative values (non-similarity)

    # we reset the index not to be out of bounds
    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)

    _ = generate_feature_similarity_dataset(
        train,
        NUM_NEG=-1,
        features=features,
        save_to=train_features_save_path
    )

    _ = generate_feature_similarity_dataset(
        test,
        NUM_NEG=-1,
        features=features,
        save_to=test_features_save_path
    )
