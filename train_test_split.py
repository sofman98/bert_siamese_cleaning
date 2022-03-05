from data_processing.file_management import load_dataset_csv, save_csv
from sklearn.model_selection import train_test_split

test_frac = 0.2
if __name__ == "__main__":
    # we load the dataset
    dataset = load_dataset_csv('datasets/cs1_us_outlets.csv')

    # we split it
    test = dataset.sample(frac=test_frac, random_state=0)
    train = dataset.drop(test.index)


    # saving the test and train sets
    save_csv(test, 'datasets/test.csv')
    save_csv(train, 'datasets/train.csv')