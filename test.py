from models.transfer_learning import load_siamese_model
from models.transfer_learning import load_embedding_model
from data_processing.file_management import load_dataset_csv, save_csv
from data_processing.data_processing import process_dataset
from data_processing.generate_similarity_dataset import generate_pair_similarity_dataset
import itertools as it
import numpy as np
import pandas as pd

# some important variables
similarity_threshold = 0.96
features = ['lat', 'lon']
num_features = len(features)

# model hyper-parameters
num_dense_layers = 3
embedding_size = 16


def predict_pair_similarity(
        model,
        test_data,
        save_to=''
    ):
    # we select the desired features
    ## numpy for faster processing
    test_features = test_data[features].to_numpy()

    print('calculating the similarity...')
    # we intialize what will be the predicted pair similarity data (id1, id2, prediction)
    predicted_pair_similarity = []

    # selecting all clusters
    unique_clusters = test_data.persistent_cluster.unique()
    for i, cluster in enumerate(unique_clusters):

        # we retrieve the indexe of the outlets present in the cluster
        cluster_indexes = test_data[test_data.persistent_cluster == cluster].index.to_numpy()
        indexes_combinatons = np.array(list(it.combinations(cluster_indexes, 2)))

        for (index1, index2) in indexes_combinatons:
            # we extract the set of features
            feature_set1 = test_features[index1]
            feature_set2 = test_features[index2]

            # adding a dimension for the prediction            
            feature_set1 = np.expand_dims(feature_set1, axis=0).astype('float32')
            feature_set2 = np.expand_dims(feature_set2, axis=0).astype('float32')

            # we predict the similarity
            prediction = model.predict([feature_set1, feature_set2])[0,0]
            
            # both (id1 == id2) and (id2 == id1) are correct, so we save them both
            ## the pair similarity dataset is also generated this way 
            predicted_pair_similarity.append([index1, index2, prediction])
            predicted_pair_similarity.append([index2, index1, prediction])

        # show progress
        print("Done: {:.2f}%".format((i+1)/len(unique_clusters)*100))
    # we save the prediction
    predicted_pair_similarity = pd.DataFrame(data=predicted_pair_similarity, columns=['outlet1', 'outlet2', 'similarity'])
    if save_to:
        print(f'Saving the feature similarity dataset to {save_to}..')
        save_csv(predicted_pair_similarity, save_to)
    
    return predicted_pair_similarity


if __name__ == "__main__":

    # we first load the model and test data
    model = load_siamese_model()
    test_data = load_dataset_csv('datasets/test.csv')
    # we process the data which normalizes the features
    test_data = process_dataset(test_data)


    # we generate our prediction
    ## if already generated with the same parameters,
    ## please use the saved one to save time
    # prediction = load_dataset_csv('predicted_similarity.csv')
    prediction = predict_pair_similarity(
        model=model,
        test_data=test_data,
        save_to='predicted_similarity.csv',
    )

    # we select the similar outlets according to the threshold
    prediction = prediction[prediction.similarity > similarity_threshold]
    print(prediction)

    # we generate the true pair similarity data
    ## this will be our ground truth
    truth = generate_pair_similarity_dataset(
        test_data,
        NUM_NEG=0  # no negative values
    )


    # now we compare between the two
    ## true positive: present in both table
    ## true negative: present in neither of table
    ## false positive: present in prediction and not in truth
    ## false negative: present in truth and not in prediction







