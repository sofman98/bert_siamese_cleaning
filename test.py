from tabnanny import verbose
from models.transfer_learning import load_siamese_model
from data_processing.file_management import load_dataset_csv
from tensorflow.math import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# some important variables
features = ['name']
num_features = len(features)
path_to_model='results/models/model_nn1.h5'
path_to_test_data='datasets/entire_feature_similarity_test.csv'

if __name__ == "__main__":

    # we first load the model and test data
    model = load_siamese_model(path_to_model)

    # we load the already generated feature similarity dataset
    test_data = load_dataset_csv(path_to_test_data).to_numpy(dtype=str)

    # we make our predictions
    X = test_data[:,:-1]
    y = test_data[:, -1].astype(int)

    print('Predicting...')
    prediction = model.predict([X[:, :num_features], X[:, num_features:]], verbose=1)
    prediction = np.round(prediction)

    # we calculate the confusion_matrix
    conf_mat = confusion_matrix(y, prediction)
    conf_mat = np.array(conf_mat)

    # we display the results
    print('Confusion Matrix:\n',conf_mat)
    ax = sns.heatmap(conf_mat, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['Negative','Positive'])
    ax.yaxis.set_ticklabels(['Negative','Positive'])
    plt.show()
    