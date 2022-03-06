from models.transfer_learning import load_siamese_model
from models.transfer_learning import load_embedding_model
from data_processing.file_management import load_dataset_csv, save_csv
from data_processing.data_processing import process_dataset
from data_processing.generate_similarity_dataset import generate_pair_similarity_dataset
import itertools as it
import numpy as np
import pandas as pd

# some important variables
similarity_threshold = 0.9
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

            # we save it            
            predicted_pair_similarity.append([index1, index2, prediction])

        # show progress
        print("Done: {:.2f}%".format((i+1)/len(unique_clusters)*100))
    # we save the prediction
    predicted_pair_similarity = pd.DataFrame(
        data=predicted_pair_similarity, 
        columns=['outlet1', 'outlet2', 'similarity'],
    )
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
    prediction = load_dataset_csv('predicted_similarity.csv')
    # print('Predicting...')
    # prediction = predict_pair_similarity(
    #     model=model,
    #     test_data=test_data,
    #     save_to='predicted_similarity.csv',
    # )

    # we select the similar outlets according to the threshold
    prediction = prediction[prediction.similarity > similarity_threshold]

    # we generate the true pair similarity data
    ## this will be our ground truth
    truth = generate_pair_similarity_dataset(
        test_data,
        NUM_NEG=-1  # all unique combinations
    )


    # now we compare between the two
    ## true positive (TP): present in both tables
    ## true negative (TN): present in neither of the tables
    ## false positive (FP): present in prediction, not present in truth
    ## false negative (FN): present in truth, not present in prediction


    # true and false positive
    print('Looking for true and false positives')
    test_id_dashmote = test_data.id_dashmote.to_numpy()
    ## for this case, we directly use test_data instead of truth
    ## reason is that it's faster (better time complexity)
    tp = 0
    fp = 0
    for index, row in prediction.iterrows():
        # we retrieve the id_dashmote of both outlets
        out1_id = test_id_dashmote[int(row.outlet1)]
        out2_id = test_id_dashmote[int(row.outlet2)]

        # we see if they have the same id_dashmote
        if out1_id == out2_id:
            tp+=1
        else:
            fp+=1


    # true and false negative
    print('Looking for true and false negatives')
    tn = 0
    fn = 0
    for index, row in truth.iterrows():
        # we retrieve the data
        outlet1 = row.outlet1
        outlet2 = row.outlet2
        similarity = row.similarity == 1 # true if 1, false if 0
        
        # we see if the pair was predicted as similar
        existing = prediction[
            ( (prediction.outlet1 == outlet1) & (prediction.outlet2 == outlet2) )
            | ( (prediction.outlet1 == outlet2) & (prediction.outlet2 == outlet1) )
        ]

        pred_similarity = len(existing) > 0 # true if exists, false if not

        # we only deal with negatives
        ## because positives were dealt with *Faster* eariler
        if (not similarity) and (not pred_similarity):
            tn+=1
        elif similarity and (not pred_similarity):
            fn+=1

        # progress display
        if index % 100 == 0:
            print("Done: {:.2f}%".format((index+1)/len(truth)*100))


    # DISPLAY THE RESULTS
    print('True Positive:, ',tp)
    print('True Negative:, ',tn)
    print('False Negative:, ',fn)
    print('False Positive:, ',fp)
    


    





        

        








