from models.difference_layer import DifferenceLayer
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import tensorflow_text

def build_embedding_model(
      inputs,
      use_encoder,
      num_dense_layers,
      embedding_size,
      preprocessor,
      encoder,
      name,
    ):
  """
  One of the submodels that create the embedding.
  Two of these will be combined to create the siamese model.
  embedding_size represents the number of nodes of the last layer.
  each layer has half the number of nodes of the preceding one.
  example with embedding_size=8 and num_dense_layers=3:
  [inputs -> 32 -> 16 -> 8].
  We name the layers for transfer learning. we separate between model A and model B.
  """
  if use_encoder:
    encoder_inputs = preprocessor(inputs)
    encoder_outputs = encoder(encoder_inputs)
    layer = encoder_outputs["pooled_output"]  # this is the resulting latent vector of shape [batch_size, 768]. 
  else:
    layer = inputs

  # we build the network according the number of dense layers and the embedding_size
  for l in range(num_dense_layers,0,-1):
    # we add a batchnormalization layer between dense ones
    layer = layers.BatchNormalization(name=f'batchNorm_{name}{num_dense_layers-l+1}')(layer)
    # dense layer
    layer = layers.Dense(embedding_size*(2**(l-1)), activation='relu', name=f'dense_{name}{num_dense_layers-l+1}')(layer)
  outputs = layer
  return Model(inputs, outputs)


def build_siamese_model(
      inputs_shape,
      num_dense_layers,
      embedding_size,
      use_encoder,
      link_to_preprocessor='',
      link_to_encoder='',
    ):
  """
  The final siamese model.
  Combines two embedding submodels then calculates 
  the absolute difference between the generated embeddings.
  The difference is then fed to a dense layer with a sigmoid
  activation to predict the probability of similarity.
  """

  if use_encoder:
    # we define the inputs with type string
    inputs_a = layers.Input(name="inputs_a", dtype=tf.string, shape = inputs_shape)
    inputs_b = layers.Input(name="inputs_b", dtype=tf.string, shape = inputs_shape)
    # we define the bert encoder and preprocessor
    preprocessor = hub.KerasLayer(link_to_preprocessor)
    encoder = hub.KerasLayer(link_to_encoder, trainable=False)
  else:
    # we define the inputs with type float (default)
    inputs_a = layers.Input(name="inputs_a", shape = inputs_shape)
    inputs_b = layers.Input(name="inputs_b", shape = inputs_shape)
    # we don't need the preporcessor or encoder
    preprocessor = None
    encoder = None

  # we add the layer we have created
  differences = DifferenceLayer()(
      build_embedding_model(
        inputs=inputs_a,
        use_encoder=use_encoder,
        num_dense_layers=num_dense_layers,
        embedding_size=embedding_size,
        preprocessor=preprocessor,
        encoder=encoder,
        name='a',
        )(inputs_a),

      build_embedding_model(
        inputs=inputs_b,
        use_encoder=use_encoder,
        num_dense_layers=num_dense_layers,
        embedding_size=embedding_size,
        preprocessor=preprocessor,
        encoder=encoder,
        name='b',
        )(inputs_b),
  )
  
  # we add the outputs layer
  outputs = layers.Dense(1, activation="sigmoid", name='outputs')(differences)

  siamese_model = Model(
      inputs=[inputs_a, inputs_b], outputs=outputs
  )
  return siamese_model