from models.difference_layer import DistanceLayer
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

def load_siamese_model(path):
  """
  Loads the trained siamese model.
  """
  model = load_model(path, custom_objects={
    'DistanceLayer': DistanceLayer,
    'KerasLayer': hub.KerasLayer
  })
  return model
