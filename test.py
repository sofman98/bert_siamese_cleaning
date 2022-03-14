<<<<<<< HEAD
from tabnanny import verbose
from models.transfer_learning import load_siamese_model
from data_processing.file_management import load_dataset_csv, save_csv, create_folder
from data_processing.generate_similarity_dataset import generate_entire_pair_similarity
from tensorflow.math import confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# some important variables
features = ['name']
num_features = len(features)
path_to_outputs_layer='results/outputs_layers/model_nn10.npy'
from_outputs_layer=True
path_to_feature_similarity_test_data='datasets/entire_feature_similarity_test.csv'
path_to_raw_test_data='datasets/test.csv'
save_prediction_to='results/predictions/last_prediction.csv'

if __name__ == "__main__":

    # we first load the model and test data
    model = load_siamese_model(
      from_outputs_layer=from_outputs_layer, 
      path=path_to_outputs_layer
    )
    
    # we load the already generated feature similarity dataset
    # if not, run train_test_split.py
    feature_similarity_test_data = load_dataset_csv(path_to_feature_similarity_test_data).to_numpy(dtype=str)
    # we make our predictions
    X = feature_similarity_test_data[:,:-1]
    y = feature_similarity_test_data[:, -1].astype(int)

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

    print('Saving the prediction...')
    # we first load the raw test set
    raw_test_data = load_dataset_csv(path_to_raw_test_data)
    # we generate the (outlet1, outlet2, similarity) dataset
    pair_similarity_combinations = generate_entire_pair_similarity(raw_test_data, 'combinations')
    # we remove the actual similarity
    pair_similarity_combinations = pair_similarity_combinations[:, :2]
    # we save the the predictions as (outlet1, outlet2, predicted similarity) 
    predicted_similarity = np.append(pair_similarity_combinations, prediction, axis=1)
    predicted_similarity = pd.DataFrame(predicted_similarity, columns=['outlet1', 'outlet2', 'similarity'], dtype=int)
    create_folder(save_prediction_to)
    save_csv(predicted_similarity, save_prediction_to)
    print(f'Saved the prediction to {save_prediction_to}')
=======
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

if __name__ == "__main__":

    # we first load the model and test data
    model = load_siamese_model(path_to_model)

    # we load the already generated feature similarity dataset
    test_data = load_dataset_csv(path_to_test_data).to_numpy()

    # we make our predictions
    X = test_data[:,:-1]
    y = test_data[:, -1]

    print('Predicting...')
    prediction = model.predict([X[:, :num_features], X[:, num_features:]])
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
>>>>>>> main
    