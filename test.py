from tabnanny import verbose
from models.transfer_learning import load_outputs_layer
from models.model_building import build_siamese_model
from data_processing.file_management import load_dataset_csv
from tensorflow.math import confusion_matrix
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# some important variables
features = ['name']
num_features = len(features)
path_to_outputs_layer='results/outputs_layers/model_nn10.npy'
path_to_test_data='datasets/entire_feature_similarity_test.csv'

# the following are for model compilation
optimizer='adam'
metric= tf.keras.metrics.Precision()

if __name__ == "__main__":

    # we first load the model and test data
    # we build the model
    model = build_siamese_model(
      inputs_shape=(),
      num_dense_layers=0,
      embedding_size=0
    )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])
    # then load the outputs layer into it
    model = load_outputs_layer(model, path_to_outputs_layer)

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
    