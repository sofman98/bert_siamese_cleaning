import tensorflow as tf
import tensorflow.keras.layers as layers

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the absolute value of the
    difference between two embeddings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, embedding1, embedding2):
        # we get the absolute value of the difference between the 2 embeddings
        abs_difference = tf.abs(tf.subtract(embedding2, embedding1))
        return abs_difference