from models.transfer_learning import load_text_embedding_model
from data_processing.file_management import load_dataset_csv, save_csv
from data_processing.generate_similarity_dataset import preprocess_dataset
import pandas as pd

# some important variables
features = ['name']
num_features = len(features)

# model hyper-parameters
path_to_train_dataset = 'datasets/train.csv'
path_to_test_dataset = 'datasets/test.csv'
save_train_embeddings_to='datasets/train_embeddings.csv'
save_test_embeddings_to='datasets/test_embeddings.csv'


if __name__ == "__main__":
    # we first load the embedding model and data
    model = load_text_embedding_model(
        num_dense_layers=0,
        embedding_size=0
    )    
    train_data = load_dataset_csv(path_to_train_dataset)
    test_data = load_dataset_csv(path_to_test_dataset)

    # we select the desired features and calculate their embedding
    ## we convert to numpy to accelerate the process
    test_features = test_data[features].to_numpy(dtype=str)
    train_features = train_data[features].to_numpy(dtype=str)

    # calculting embeddings
    train_embeddings = model.predict(train_features)
    test_embeddings = model.predict(test_features)

    # saving the embeddings in a CSV file
    train_embeddings = pd.DataFrame(train_embeddings)
    test_embeddings = pd.DataFrame(test_embeddings)
    save_csv(train_embeddings, save_train_embeddings_to)
    save_csv(test_embeddings, save_test_embeddings_to)
