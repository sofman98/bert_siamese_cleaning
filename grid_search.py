from pickletools import optimize
from data_processing.file_management import load_dataset
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
from data_processing.data_processing import process_dataset
from models.models import build_siamese_network
from models.difference_layer import DifferenceLayer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
tf.config.experimental_run_functions_eagerly(True) # we need this because of the custom layer

if __name__ == "__main__":
  # we declare some important variables
  NUM_NEG = 1 #dataset is balanced
  features = ['lat', 'lon']
  num_features = len(features)
  save_feature_similarity_dataset_to='datasets/feature_similarity_dataset.parquet.gzip'
  num_epochs = 100
  training_batch_size = 64

  # DATASET PREPARATION

  # we load the raw data file
  dataset = load_dataset(path = 'datasets/cs1_us_outlets.parquet.gzip')
  # we generate the feature similarity and save it (feature_set1, feature_set2, similarity)
  feature_similarity_dataset = generate_feature_similarity_dataset(
    dataset,
    features=features,
    NUM_NEG=NUM_NEG,
    save_to=save_feature_similarity_dataset_to
  )
  feature_similarity_dataset = feature_similarity_dataset.to_numpy()
  # we split the dataset into train and test
  X = feature_similarity_dataset[:, :-1]
  y = feature_similarity_dataset[:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


  # HYPER-PARAMETER RANGES (for tuning)
  range_num_dense_layers = [1, 2, 3] 
  range_embedding_size = [4, 8, 16]
  range_optimizer = ['adam']

  # Callbacks
  ## early stopping
  es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
  )
  ## checkpoint
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='temp_model_delete_me.h5',
    save_best_only=True,
    verbose=1,
    monitor='val_accuracy', 
    mode='max'
  )

  # create a file for saving the best registered results
  with open('grid_search_results.csv', 'w') as file:
    file.write('num_dense_layers,embedding_size,optimizer,loss,accuracy\n')
  

  ########   GRID SEARCH   ########

  for num_dense_layers in range_num_dense_layers:
    for embedding_size in range_embedding_size:
      for optimizer in range_optimizer:

        # Siamese model building
        model = build_siamese_network(
            input_shape=num_features,
            num_dense_layers=num_dense_layers,
            embedding_size=embedding_size
        )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
        model = load_model('temp_model_delete_me.h5', custom_objects={'DifferenceLayer': DifferenceLayer})
        ## calculate the loss & accuracy
        loss, accuracy = model.evaluate([X_test[:, :num_features], X_test[:, num_features:]], y_test)
        ## then save the results along with the architecture for later
        with open('grid_search_results.csv', 'a') as file:
          file.write(f'{num_dense_layers},{embedding_size},{optimizer},{loss},{accuracy}\n')

