from pickletools import optimize
from data_processing.file_management import load_dataset
from data_processing.generate_similarity_dataset import generate_feature_similarity_dataset
from data_processing.data_processing import process_dataset
from models.models import build_siamese_network
tf.config.experimental_run_functions_eagerly(True) # we need this because of the custom layer

if __name__ == "__main__":
  # we declare some important information
  NUM_NEG = 1 #dataset is balanced
  features = ['lat', 'lon']
  num_features = len(features)
  num_dense_layers = 2
  embedding_size = 4
  optimizer = 'adam'
  
  # we load the raw data file
  dataset = load_dataset(path = './cs1_us_outlets.parquet.gzip')
  dataset = process_dataset(dataset)

  # we generate the feature similarity and save it (feature_set1, feature_set2, similarity)
  feature_similarity_dataset = generate_feature_similarity_dataset(
    dataset,
    features=features,
    NUM_NEG=1,
    save_to='feature_similarity_dataset.parquet'
  )
  model = build_siamese_network()
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  model.fit(
      [X_train[:, :num_features], X_train[:, num_features:]], y_train, 
      epochs=100,
      batch_size=64,
      verbose=2,
      validation_data=([X_test[:, :num_features], X_test[:, num_features:]], y_test)
    )

  
  print(feature_similarity_dataset)