from os.path import exists
from data_processing.file_management import load_dataset_csv, create_folder
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
from models.model_building import build_siamese_model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# training parameters
metric = tf.keras.metrics.Precision()
num_epochs = 10
training_batch_size = 64
early_stopping_patience = 20

# path for loading data
path_to_train_data = 'datasets/feature_similarity_train.npy'

# path for saving data 
results_save_path = f'results/grid_search_results/results.csv'
last_model_save_path = 'results/models/last_trained_model.h5'
outputs_layer_save_path = 'results/outputs_layers/last_trained_model.npy'


# HYPER-PARAMETER RANGES (for tuning)
# if you use different values than 0, you need to load your entire model when testing
range_num_dense_layers = [3] 
range_embedding_size = [8]
range_optimizer = ['adam']

if __name__ == "__main__":

  # DATASET LOADING
  feature_similarity_dataset = np.load(path_to_train_data, allow_pickle=True)
  
  # we split the dataset into train and test
  X = feature_similarity_dataset[:, :-1]
  y = feature_similarity_dataset[:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

  # Callbacks
  ## early stopping
  es_callback = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{metric.name}',
    patience=early_stopping_patience,
    mode='max'
  )
  ## checkpoint
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=last_model_save_path,
    save_best_only=True,
    verbose=1,
    monitor=f'val_{metric.name}',
    mode='max'
  )
  
  ######## FILE MANAGEMENT ########

  # we create the folders if they don't exist
  create_folder(results_save_path)
  create_folder(last_model_save_path)
  create_folder(outputs_layer_save_path)

  # create a file for saving the best registered results
  ## only if file doesn't exist
  if not exists(results_save_path):
    with open(results_save_path, 'w') as file:
        file.write(f"num_dense_layers,embedding_size,optimizer,loss,{metric.name}\n") # title
  

  ########   GRID SEARCH   ########

  for num_dense_layers in range_num_dense_layers:
    for embedding_size in range_embedding_size:
      for optimizer in range_optimizer:

        # we get the number of features for each input
        num_features = int(X.shape[1] / 2)

        # Siamese model building
        model = build_siamese_model(
            inputs_shape=(num_features,), #for text
            num_dense_layers=num_dense_layers,
            embedding_size=embedding_size,
            use_encoder=False
        )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])

        # Train
        hist = model.fit(
            [X_train[:, :num_features], X_train[:, num_features:]], y_train, 
            epochs=num_epochs,
            batch_size=training_batch_size,
            verbose=1,
            validation_data=([X_test[:, num_features:], X_test[:, :num_features]], y_test),
            callbacks=[es_callback, cp_callback]
        )

        # when training is over
        # we get the epoch with the best metric score
        best_epoch = np.argmax(hist.history[f'val_{metric.name}'])
        metric_score = hist.history[f'val_{metric.name}'][best_epoch]
        loss = hist.history['val_loss'][best_epoch]
        ## then save the results along with the architecture for later
        with open(results_save_path, 'a') as file:
          file.write(f'{num_dense_layers},{embedding_size},{optimizer},{loss},{metric_score}\n')
        
