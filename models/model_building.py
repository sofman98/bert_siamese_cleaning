from models.difference_layer import DifferenceLayer
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import tensorflow_text

def build_text_embedding_model(
      inputs,
      preprocessor,
      encoder,
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
<<<<<<< HEAD
  [inputs -> 32 -> 16 -> 8].
  We name the layers for transfer learning. we separate between model A and model B.
  """
  encoder_inputs = preprocessor(inputs)
  encoder_outputs = encoder(encoder_inputs)
  layer = encoder_outputs["pooled_output"]  # this is the resulting latent vector of shape [batch_size, 768]. 

  # we build the network according the number of dense layers and the embedding_size
  for l in range(num_dense_layers,0,-1):
    # we add a batchnormalization layer between dense ones
    layer = layers.BatchNormalization(name=f'batchNorm_{name}{num_dense_layers-l+1}')(layer)
    # dense layer
    layer = layers.Dense(embedding_size*(2**(l-1)), activation='relu', name=f'dense_{name}{num_dense_layers-l+1}')(layer)
  outputs = layer
  return Model(inputs, outputs)

def build_embedding_model(
    inputs,
    num_dense_layers,
    embedding_size,
    name,
  ):
  """
  One of the submodels that create the embedding FOR TEXTS using a pretrained BERT model.
  Two of these will be combined to create the siamese model.
  embedding_size represents the number of nodes of the last layer.
  each layer has half the number of nodes of the preceding one.
  example with embedding_size=8 and num_dense_layers=3:
  [inputs -> 64 -> 32 -> 8].
=======
  [input -> 32 -> 16 -> 8].
>>>>>>> main
  We name the layers for transfer learning. we separate between model A and model B.
  """
  layer = inputs

  # we build the network according the number of dense layers and the embedding_size
  for l in range(num_dense_layers,0,-1):
    layer = layers.Dense(embedding_size*(2**(l-1)), activation='relu', name=f'dense_{name}{num_dense_layers-l+1}')(layer)
    # we add a batchnormalization layer between dense ones
    if l!=1:  #if not the last layer (embedding)
      layer = layers.BatchNormalization(name=f'batchNorm_{name}{num_dense_layers-l+1}')(layer)

  outputs = layer
  return Model(inputs, outputs)

def build_siamese_model(
      inputs_shape,
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

  # we define the inputs
  inputs_a = layers.Input(name="inputs_a", dtype=tf.string, shape = inputs_shape)
  inputs_b = layers.Input(name="inputs_b", dtype=tf.string, shape = inputs_shape)

  # we define the bert encoder and preprocessor
  preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
  encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/3", trainable=False)

  # we add the layer we have created
  differences = DifferenceLayer()(
      build_text_embedding_model(
        inputs_a,
        preprocessor,
        encoder,
        num_dense_layers,
        embedding_size,
        name='a',
        )(inputs_a),

      build_text_embedding_model(
        inputs_b,
        preprocessor,
        encoder,
        num_dense_layers,
        embedding_size,
        name='b',
        )(inputs_b),
  )
  
  # we add the outputs layer
  outputs = layers.Dense(1, activation="sigmoid", name='outputs')(differences)

  siamese_model = Model(
      inputs=[inputs_a, inputs_b], outputs=outputs
  )
  return siamese_model