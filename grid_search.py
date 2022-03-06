from data_processing.file_management import load_dataset_csv
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
from models.model_building import build_siamese_model
from models.transfer_learning import load_siamese_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.config.experimental_run_functions_eagerly(True) # we need this because of the custom layer

# we declare some important variables
NUM_NEG = 10
metric_name = 'precision' 
metric = tf.keras.metrics.Precision()
features = ['lat', 'lon']   # selected features
num_features = len(features)
num_epochs = 100
training_batch_size = 64
early_stopping_patience = 20
save_feature_similarity_dataset_to=''  # no saving if empty
# path for saving data 
last_model_save_path = 'results/models/last_trained_model.h5'
results_save_path = f'results/grid_search_results_num_neg_{NUM_NEG}_{metric_name}.csv'


# HYPER-PARAMETER RANGES (for tuning)
range_num_dense_layers = [1, 2, 3] 
range_embedding_size = [8, 16, 32]
range_optimizer = ['adam', 'sgd']

if __name__ == "__main__":

  # DATASET PREPARATION

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

  # ..or directly load a pre-computed feature_dataset this way:
  # feature_similarity_dataset = load_dataset_csv('datasets/entire_feature_similarity_train.csv')
  # this one above contains all negative instances
  # you can try generating one with diffrent NUM_NEG values

  
  # we split the dataset into train and test
  feature_similarity_dataset = feature_similarity_dataset.to_numpy()
  X = feature_similarity_dataset[:, :-1]
  y = feature_similarity_dataset[:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

  # Callbacks
  ## early stopping
  es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
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

  # create a file for saving the best registered results
  with open(results_save_path, 'w') as file:
    file.write(f'num_dense_layers,embedding_size,optimizer,loss,{metric_name}\n')
  

  ########   GRID SEARCH   ########

  for num_dense_layers in range_num_dense_layers:
    for embedding_size in range_embedding_size:
      for optimizer in range_optimizer:

        # Siamese model building
        model = build_siamese_model(
            input_shape=num_features,
            num_dense_layers=num_dense_layers,
            embedding_size=embedding_size
        )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])

        # Train
        model.fit(
            [X_train[:, :num_features], X_train[:, num_features:]], y_train, 
            epochs=num_epochs,
            batch_size=training_batch_size,
            verbose=2,
            validation_data=([X_test[:, :num_features], X_test[:, num_features:]], y_test),
            callbacks=[es_callback, cp_callback]
        )

        # when training is over
        ## load the best weights for this set of hyperparameters
        model = load_siamese_model(last_model_save_path)
        ## calculate the loss & metric score
        loss, metric_score = model.evaluate([X_test[:, :num_features], X_test[:, num_features:]], y_test)
        ## then save the results along with the architecture for later
        with open(results_save_path, 'a') as file:
          file.write(f'{num_dense_layers},{embedding_size},{optimizer},{loss},{metric_score}\n')

