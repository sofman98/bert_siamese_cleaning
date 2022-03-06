from models.transfer_learning import load_siamese_model
from data_processing.file_management import load_dataset_csv
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
from tensorflow.math import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# some important variables
features = ['lat', 'lon']
num_features = len(features)

# model hyper-parameters
num_dense_layers = 3
embedding_size = 16


if __name__ == "__main__":

    # we first load the model and test data
    model = load_siamese_model()
    test_data = load_dataset_csv('datasets/test.csv')

    # generate the feature similarity dataset (feature_set1, feature_set2, similarity)
    # NUM_NEG==-1 which means it includes all possible unique combinations
    # for negative values (non-similarity)
    # test_data = generate_feature_similarity_dataset(
    #     test_data,
    #     NUM_NEG=-1,
    #     features=features,
    #     save_to='datasets/entire_feature_similarity_test.csv'
    # )
    # ..or load it if already generated
    test_data = load_dataset_csv('datasets/entire_feature_similarity_test.csv')

    # we make our predictions

    print('Predicting...')
    X = test_data.to_numpy()[:,:-1]
    y = test_data.to_numpy()[:, -1]
    prediction = model.predict([X[:, num_features:], X[:, :num_features]])

    # we calculate the confusion_matrix
    confusion_matrix = confusion_matrix(y, prediction)
    confusion_matrix = np.array(confusion_matrix)

    # we display the results
    print('Confusion Matrix:\n',confusion_matrix)
    ax = sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Negative','Positive'])
    ax.yaxis.set_ticklabels(['Negative','Positive'])
    plt.show()



        

        








