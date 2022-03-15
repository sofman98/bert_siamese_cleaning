from data_processing.file_management import load_dataset_csv
from data_processing.data_processing import filter_features
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import numpy as np

# some important variables
feature = 'name'
save_embeddings_to = 'datasets/embeddings.npy'
path_to_dataset = 'datasets/cs1_us_outlets.csv'

# encoder info
link_to_preprocessor = "http://tfhub.dev/tensorflow/albert_en_preprocess/3"
link_to_encoder = "https://tfhub.dev/tensorflow/albert_en_base/3"

if __name__ == "__main__":

  # we first load the data
  dataset = load_dataset_csv(path_to_dataset)

  # we select the features
  ## we convert to numpy to accelerate the process
  dataset = filter_features(dataset, [feature]).to_numpy(dtype=str)

  ### MODEL BUILDING ###
  # we define the inputs
  inputs = layers.Input(name="inputs", dtype='string', shape = ())
  # we define the bert encoder and preprocessor
  preprocessor = hub.KerasLayer(link_to_preprocessor)(inputs)
  encoder = hub.KerasLayer(link_to_encoder, trainable=False)(preprocessor)

  # we build the model
  model = Model(inputs, encoder)

  # calculting embeddings
  print('generating the embeddings...')
  embeddings = model.predict(dataset, verbose=1)

  # saving the embeddings in a npy file
  np.save(save_embeddings_to, embeddings)

  print(f'Embeddings saved to {save_embeddings_to}.')
