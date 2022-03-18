import tensorflow as tf
import numpy as np
from os.path import exists
from data_processing.file_management import create_folder
from sklearn.model_selection import train_test_split
from models.model_building import build_siamese_model
from tensorflow.keras.metrics import Precision, Recall

# training parameters
metrics = [Precision(), Recall()]
num_epochs = 100
training_batch_size = 64
early_stopping_patience = 20

# path for loading data
path_to_train_data = 'datasets/feature_similarity_train.npy'

# path for saving data 
results_save_path = 'results/grid_search_results/results.csv'
last_model_save_path = 'results/models/last_trained_model.h5'

# HYPER-PARAMETER RANGES (for tuning)
range_num_dense_layers = [0] 
range_embedding_size = [0]
range_optimizer = ['adam']

if __name__ == "__main__":

  # TRAIN-SET LOADING
  feature_similarity_dataset = np.load(path_to_train_data, allow_pickle=True)
  
  # We split the dataset into train and test
  X = feature_similarity_dataset[:, :-1]
  y = feature_similarity_dataset[:, -1]
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

  # Callbacks
  ## early stopping
  es_callback = tf.keras.callbacks.EarlyStopping(
    patience=early_stopping_patience,
    monitor='val_loss',
    mode='min'
  )
  ## checkpoint
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=last_model_save_path,
    save_best_only=True,
    verbose=1,
    monitor='val_loss',
    mode='min'
  )
  
  ######## FILE MANAGEMENT ########

  # we create the folders if they don't exist
  create_folder(results_save_path)
  create_folder(last_model_save_path)

  # create a file for saving the best registered results
  ## only if file doesn't exist
  if not exists(results_save_path):
    with open(results_save_path, 'w') as file:
        file.write(f"num_dense_layers,embedding_size,optimizer,loss,{metrics[0].name},{metrics[1].name}\n") # title
  

  ########   GRID SEARCH   ########

  for num_dense_layers in range_num_dense_layers:
    for embedding_size in range_embedding_size:
      for optimizer in range_optimizer:

        # we get the number of features for each input
        num_features = int(X.shape[1] / 2)

        # Siamese model building
        model = build_siamese_model(
            inputs_shape=(num_features,),
            num_dense_layers=num_dense_layers,
            embedding_size=embedding_size,
            use_encoder=False
        )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

        # Train
        hist = model.fit(
            [X_train[:, :num_features], X_train[:, num_features:]], y_train, 
            epochs=num_epochs,
            batch_size=training_batch_size,
            verbose=1,
            validation_data=([X_val[:, num_features:], X_val[:, :num_features]], y_val),
            callbacks=[es_callback, cp_callback]
        )

        # when training is over
        # we get the epoch with the best metric score
        loss = np.min(hist.history['val_loss'])
        metric_scores = [np.max(hist.history[f'val_{metric.name}']) for metric in metrics]
        ## then save the results along with the architecture for later
        with open(results_save_path, 'a') as file:
          file.write(f'{num_dense_layers},{embedding_size},{optimizer},{loss},{metric_scores[0]},{metric_scores[1]}\n')
        
