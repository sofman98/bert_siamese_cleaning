from models.transfer_learning import load_siamese_model
from data_processing.file_management import load_dataset_csv
from tensorflow.math import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# some important variables
features = ['lat', 'lon']
num_features = len(features)
path_to_model='results/models/best_model.h5'
path_to_test_data='datasets/entire_feature_similarity_test.csv'

# model hyper-parameters
num_dense_layers = 3
embedding_size = 16


if __name__ == "__main__":

    # we first load the model and test data
    model = load_siamese_model(path_to_model)

    # we load the already generated feature similarity dataset
    test_data = load_dataset_csv(path_to_test_data).to_numpy()

    # we make our predictions
    print('Predicting...')
    X = test_data[:,:-1]
    y = test_data[:, -1]

    prediction = model.predict([X[:, num_features:], X[:, :num_features]])
    prediction = np.round(prediction)

    # we calculate the confusion_matrix
    confusion_matrix = confusion_matrix(y, prediction)
    confusion_matrix = np.array(confusion_matrix)

    # we display the results
    print('Confusion Matrix:\n',confusion_matrix)
    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Negative','Positive'])
    ax.yaxis.set_ticklabels(['Negative','Positive'])
    plt.show()


        

        








