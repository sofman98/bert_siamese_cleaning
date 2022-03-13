from os.path import exists
from data_processing.file_management import load_dataset_csv
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
from data_processing.file_management import create_folder
from models.model_building import build_siamese_model
from models.transfer_learning import load_siamese_model, save_outputs_layer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# tf.config.experimental_run_functions_eagerly(True) # we need this because of the custom layer
tf.config.run_functions_eagerly(True)

# we declare some important variables
NUM_NEG = 0 # number of negative instances per outlet - Dataset is balanced if NUM_NEG==1
all_neg_combinations = False # if True then NUM_NEG is useless, uses all negative instance, if False then considers NUM_NEG
metric_name = 'precision' 
metric = tf.keras.metrics.Precision()
features = ['name']   # selected features
num_features = len(features)
num_epochs = 1
training_batch_size = 12 # big batch size might cause ram overload
early_stopping_patience = 20
save_feature_similarity_dataset_to=''  # no saving if empty
# path for saving data 
results_save_path = f'results/grid_search_results_num_neg_{NUM_NEG}_{metric_name}.csv'
last_model_save_path = 'results/models/last_trained_model.h5'
outputs_layer_save_path = 'results/outputs_layers/last_trained_model.npy'


# HYPER-PARAMETER RANGES (for tuning)
range_num_dense_layers = [0] 
range_embedding_size = [0]
range_optimizer = ['adam']

if __name__ == "__main__":

  # DATASET PREPARATION
  if all_neg_combinations:
    # ..or directly load a pre-computed feature_dataset this way:
    feature_similarity_dataset = load_dataset_csv('datasets/entire_feature_similarity_train.csv')
    # this one above contains all negative instances
    # you can try generating one with diffrent NUM_NEG values
  else:
    # we load the raw data file
    dataset = load_dataset_csv(path = 'datasets/train.csv')
    # we generate the feature similarity and save it (feature_set1, feature_set2, similarity)
    feature_similarity_dataset = generate_feature_similarity_dataset(
      dataset,
      features=features,
      NUM_NEG=NUM_NEG,
      all_neg_combinations=False,
      save_to=save_feature_similarity_dataset_to
    )
  
  # we split the dataset into train and test
  feature_similarity_dataset = feature_similarity_dataset.to_numpy(dtype=str)
  X = feature_similarity_dataset[:, :-1]
  y = feature_similarity_dataset[:, -1].astype(int)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

  # Callbacks
  ## early stopping
  es_callback = tf.keras.callbacks.EarlyStopping(
    monitor=f'val_{metric_name}',
    patience=early_stopping_patience,
    mode='max'
  )
  ## checkpoint
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=last_model_save_path,
    save_best_only=True,
    verbose=1,
    monitor=f'val_{metric_name}',
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
        file.write(f"num_dense_layers,embedding_size,optimizer,loss,{metric_name}\n")
  

  ########   GRID SEARCH   ########

  for num_dense_layers in range_num_dense_layers:
    for embedding_size in range_embedding_size:
      for optimizer in range_optimizer:

        # Siamese model building
        model = build_siamese_model(
            inputs_shape=(), #for text
            num_dense_layers=num_dense_layers,
            embedding_size=embedding_size
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
        best_epoch = np.argmax(hist.history[f'val_{metric_name}'])
        metric_score = hist.history[f'val_{metric_name}'][best_epoch]
        loss = hist.history['val_loss'][best_epoch]
        ## then save the results along with the architecture for later
        with open(results_save_path, 'a') as file:
          file.write(f'{num_dense_layers},{embedding_size},{optimizer},{loss},{metric_score}\n')
          
        # as the model is very big, we only save the output layer
        model = load_siamese_model(last_model_save_path)
        save_outputs_layer(model, outputs_layer_save_path)
        
