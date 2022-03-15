from data_processing.file_management import create_folder
from models.model_building import build_text_embedding_model, build_siamese_model
import tensorflow.keras.layers as layers
from models.difference_layer import DifferenceLayer
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf


# for loading the siamese model

def load_siamese_model(
    path='',
    optimizer='adam',
    metric=tf.keras.metrics.Precision()
  ):
  """
  For loading the model while indicating the custom layer we made.
  """
  if not path:
    path = 'results/models/best_model.h5'
  model = load_model(path, custom_objects={
    'DifferenceLayer': DifferenceLayer,
    'KerasLayer': hub.KerasLayer
  })
  return model


# for loading a submodel

def load_text_embedding_model(
  num_dense_layers,
  embedding_size,
  from_outputs_layer,
  use_encoder,
  path_to_siamese,
  ):
  """
  Loads the siamese model and extracts the trained embedding model.
  """
  siamese_model = load_siamese_model(from_outputs_layer, path_to_siamese)

  # we get the inputs shape of one submodel from that of the siamese model
  inputs_shape = siamese_model.inputs_shape[0][1:]
  inputs_a = layers.inputs(name="inputs_a", shape = inputs_shape)

  # we build the model
  embedding_model = build_text_embedding_model(
      inputs=inputs_a,
      use_encoder=use_encoder,
      num_dense_layers=num_dense_layers,
      embedding_size=embedding_size,
      name='a'
  )

  # we load the weights into the model by name
  embedding_model.load_weights(path_to_siamese, by_name=True)

  return embedding_model