from models.transfer_learning import load_embedding_model
from data_processing.file_management import load_dataset_csv, save_csv
from data_processing.generate_similarity_dataset import preprocess_dataset
import pandas as pd

# some important variables
features = ['name']

# model hyper-parameters
num_dense_layers = 3
embedding_size = 16
path_to_test_dataset = 'datasets/test.csv'
save_embeddings_to='datasets/test_embeddings.csv'


if __name__ == "__main__":
    # we first load the embedding model and test data
    model = load_embedding_model(
        num_dense_layers=num_dense_layers,
        embedding_size=embedding_size
    )    
    test_data = load_dataset_csv(path_to_test_dataset)

    # we process the data which normalizes the features
    test_data = preprocess_dataset(test_data)

    # we select the desired features and calculate their embedding
    ## we convert to numpy to accelerate the process
    test_features = test_data[features].to_numpy()

    # calculting embeddings
    embeddings = model.predict(test_features)

    # saving the embeddings in a CSV file
    embeddings = pd.DataFrame(embeddings)
    print(embeddings)
    save_csv(embeddings, save_embeddings_to)

