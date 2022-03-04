from models.difference_layer import DifferenceLayer
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

class SiameseModel(Model):
  """
  The siamese model we train.
  """
  def __init__(
      self,
      input_shape,
      num_dense_layers,
      embedding_size,
    ):
    super().__init__()
    self.input_shape = input_shape
    self.num_dense_layers = num_dense_layers
    self.embedding_size = embedding_size

  def call(self, inputs):
    self._build_siamese_network()

  def _build_embedding_network(self, input):
    """
    One of the submodels that create the embedding.
    Two of these will be combined to create the siamese model.  
    """
    layer = input

    # we build the network according the number of dense layers and the embedding_size
    for l in range(self.num_dense_layers,0,-1):
      layer = layers.Dense(self.embedding_size*(2**(l-1)), activation='relu')(layer)
      layer = layers.BatchNormalization()(layer)

    output = layer
    return Model(input, output)

  def _build_siamese_network(self):
    """
    The final siamese model.
    Combines two embedding submodels then calculates 
    the absolute difference between the generated embeddings.
    The difference is then fed to a dense layer with a sigmoid
    activation to predict the probability of similarity.
    """

    # we define the input
    input1 = layers.Input(name="input_1", shape = self.input_shape)
    input2 = layers.Input(name="input_2", shape = self.input_shape)

    # we add the layer we have created
    differences = DifferenceLayer()(
        self._build_embedding_network(input1)(input1),
        self._build_embedding_network(input2)(input2),
    )
    
    # we add the output layer
    output = layers.Dense(1, activation="sigmoid")(differences)

    siamese_network = Model(
        inputs=[input1, input2], outputs=output
    )
    return siamese_network