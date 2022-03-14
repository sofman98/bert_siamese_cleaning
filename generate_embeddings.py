from sklearn import datasets
from models.model_building import build_text_embedding_model
from data_processing.file_management import load_dataset_csv, save_csv
from data_processing.generate_similarity_dataset import preprocess_dataset
import pandas as pd
import tensorflow.keras.layers as layers
import tensorflow_hub as hub
import numpy as np

# some important variables
features = ['name']
num_features = len(features)

# model hyper-parameters
path_to_dataset = 'datasets/cs1_us_outlets.csv'
save_embeddings_to ='datasets/embeddings.npy'

# encoder info
link_to_preprocessor = "http://tfhub.dev/tensorflow/albert_en_preprocess/3"
link_to_encoder = "https://tfhub.dev/tensorflow/albert_en_base/3"

if __name__ == "__main__":
  # we first load the embedding model and data
  # we define the inputs
  inputs_a = layers.Input(name="inputs_a", dtype='string', shape = ())
  # we define the bert encoder and preprocessor
  preprocessor = hub.KerasLayer(link_to_preprocessor)
  encoder = hub.KerasLayer(link_to_encoder, trainable=False)

  model = build_text_embedding_model(
    inputs_a,
    preprocessor,
    encoder,
    num_dense_layers=0,
    embedding_size=0,
    name='a',
  )    
  dataset = load_dataset_csv(path_to_dataset)

  # we select the desired features and calculate their embedding
  ## we convert to numpy to accelerate the process
  dataset_features = dataset[features].to_numpy(dtype=str)

  # calculting embeddings
  print('generating the embeddings...')
  embeddings = model.predict(dataset_features, verbose=1)

  # saving the embeddings in a npy file
  np.save(save_embeddings_to, embeddings)

  print(f'Embeddings saved to {save_embeddings_to}')
