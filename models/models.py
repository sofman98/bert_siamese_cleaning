from models.difference_layer import DifferenceLayer
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

def build_embedding_network(
      input,
      num_dense_layers,
      embedding_size
    ):
  """
  One of the submodels that create the embedding.
  Two of these will be combined to create the siamese model.
  embedding_size represents the number of nodes of the last layer.
  each layer has half the number of nodes of the preceding one.
  example with embedding_size=8 and num_dense_layers=3:
  [input -> 64 -> 32 -> 8]
  """
  layer = input

  # we build the network according the number of dense layers and the embedding_size
  for l in range(num_dense_layers,0,-1):
    layer = layers.Dense(embedding_size*(2**(l-1)), activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)

  output = layer
  return Model(input, output)

def build_siamese_network(
      input_shape,
      num_dense_layers,
      embedding_size
    ):
  """
  The final siamese model.
  Combines two embedding submodels then calculates 
  the absolute difference between the generated embeddings.
  The difference is then fed to a dense layer with a sigmoid
  activation to predict the probability of similarity.
  """

  # we define the input
  input1 = layers.Input(name="input_1", shape = input_shape)
  input2 = layers.Input(name="input_2", shape = input_shape)

  # we add the layer we have created
  differences = DifferenceLayer()(
      build_embedding_network(
        input1,
        num_dense_layers,
        embedding_size,
        )(input1),

      build_embedding_network(
        input2,
        num_dense_layers,
        embedding_size,
        )(input2),
  )
  
  # we add the output layer
  output = layers.Dense(1, activation="sigmoid")(differences)

  siamese_network = Model(
      inputs=[input1, input2], outputs=output
  )
  return siamese_network


