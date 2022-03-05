import pandas as pd
from data_processing.file_management import load_dataset_csv
from data_processing.generate_similarity_dataset import process_dataset
from sklearn.cluster import KMeans
import itertools as it
import numpy as np

if __name__ == "__main__":
    # we first load the test data
    test_data = load_dataset_csv('datasets/test.csv')
    # we process the data which normalizes the features
    test_data = process_dataset(test_data)

    # we load the embeddings
    ## numpy for faster processing
    test_embeddings = load_dataset_csv('datasets/test_embeddings.csv').to_numpy()

    # we group the test data into clusters
    # selecting all clusters
    unique_clusters = test_data.persistent_cluster.unique()

    for cluster in unique_clusters:
        # we retrieve the indexe of the outlets present in the cluster
        cluster_indexes = test_data[test_data.persistent_cluster == cluster].index.to_numpy()
        indexes_combinatons = np.array(list(it.combinations(cluster_indexes, 2)))

        distances = []
        for (outlet1, outlet2) in indexes_combinatons:
            # we select one entity to compare to the rest
            outlet1_embeddings = test_embeddings[outlet1]

            # we select a second entity and calculate the distance between the two
            ## multiple techniques exist for distance calculations
            ## but we simpy use the absolute difference
            candidate_embeddings = test_embeddings[candidate]
            distances.append(np.absolute(outlet1_embeddings-candidate_embeddings))
        
        # we have the calculated distances
        distances = np.array(distances)

        # we use kmeans with k=2
        ## to separate between the closest and furthest instances
        kmeans = KMeans(n_clusters=2, random_state=0).fit(distances)

        # we select the cluster in which outlet1 fell
        ## as its distance from itsel is zero
        labels = kmeans.labels_
        closest_cluster = kmeans.labels_[i]
        print(cluster_indexes)
            print(labels)

            selected_indexes = np.where(labels == closest_cluster)
            cluster_indexes[selected_indexes]

            break

        break
