from models.difference_layer import DifferenceLayer
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

def build_embedding_model(
      input,
      num_dense_layers,
      embedding_size,
      name,
    ):
  """
  One of the submodels that create the embedding.
  Two of these will be combined to create the siamese model.
  embedding_size represents the number of nodes of the last layer.
  each layer has half the number of nodes of the preceding one.
  example with embedding_size=8 and num_dense_layers=3:
  [input -> 64 -> 32 -> 8].
  We name the layers for transfer learning. we separate between model A and model B.
  """
  layer = input

  # we build the network according the number of dense layers and the embedding_size
  for l in range(num_dense_layers,0,-1):
    layer = layers.Dense(embedding_size*(2**(l-1)), activation='relu', name=f'dense_{name}{num_dense_layers-l+1}')(layer)
    # we add a batchnormalization layer between dense ones
    if l!=1:  #if not the last layer (embedding)
      layer = layers.BatchNormalization(name=f'batchNorm_{name}{num_dense_layers-l+1}')(layer)

  output = layer
  return Model(input, output)

def build_siamese_model(
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
  input_a = layers.Input(name="input_a", shape = input_shape)
  input_b = layers.Input(name="input_b", shape = input_shape)

  # we add the layer we have created
  differences = DifferenceLayer()(
      build_embedding_model(
        input_a,
        num_dense_layers,
        embedding_size,
        name='a',
        )(input_a),

      build_embedding_model(
        input_b,
        num_dense_layers,
        embedding_size,
        name='b',
        )(input_b),
  )
  
  # we add the output layer
  output = layers.Dense(1, activation="sigmoid", name='output')(differences)

  siamese_model = Model(
      inputs=[input_a, input_b], outputs=output
  )
  return siamese_model


