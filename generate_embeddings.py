from data_processing.file_management import load_csv
from data_processing.data_processing import filter_features
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import tensorflow_text
import numpy as np

# some important variables
feature = 'name'
save_embeddings_to = 'datasets/embeddings.npy'
path_to_dataset = 'datasets/cs1_us_outlets.csv'

# encoder info
link_to_preprocessor = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
link_to_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"

if __name__ == "__main__":

  # we first load the data
  dataset = load_csv(path_to_dataset)

  # we select the features
  ## we convert to numpy to accelerate the process
  dataset = filter_features(dataset, [feature]).to_numpy(dtype=str)

  ### MODEL BUILDING ###
  # we define the encoder and preprocessor
  preprocessor = hub.KerasLayer(link_to_preprocessor)
  encoder = hub.KerasLayer(link_to_encoder, trainable=False)

  # we start building the model
  inputs = layers.Input(name="inputs", dtype='string', shape = ())
  encoder_inputs = preprocessor(inputs)
  encoder_outputs = encoder(encoder_inputs)
  outputs = encoder_outputs["pooled_output"]
  # we build the model
  model = Model(inputs, outputs)

  # calculting embeddings
  print('generating the embeddings...')
  embeddings = model.predict(dataset, verbose=1)


  # saving the embeddings in a npy file
  np.save(save_embeddings_to, embeddings)

  print(f'embeddings saved to {save_embeddings_to}.')
