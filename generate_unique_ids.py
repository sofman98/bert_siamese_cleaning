from enum import unique
from json import load
import numpy as np

from data_processing.file_management import load_dataset_csv


if __name__ == "__main__":

  prediction = load_dataset_csv('results/prediction.csv')
  test = load_dataset_csv('datasets/test.csv')
  
  duplicates_indexes = np.where(prediction[:,2] > 0.5)

  prediction = prediction[duplicates_indexes]

  # we sort the predictions to give priority
  # to the most probable duplicates
  prediction = np.sort(prediction, axis=2)

  duplicates = prediction[:, :2]

  counter = 0
  id_dashmotes = np.empty((test.shape[0], 1))
  for (outlet1, outlet2) in duplicates:
    # if none of the outlets was attributed an id_dashmote
    if not id_dashmotes[outlet1] and not id_dashmotes[outlet2]:
      id_dashmotes[outlet1] = counter
      id_dashmotes[outlet2] = counter
      counter+=1

    # if only one of them was attributed one id_dashmote
    elif id_dashmotes[outlet1] and not id_dashmotes[outlet2]:
      id_dashmotes[outlet2] = id_dashmotes[outlet1]

    elif not id_dashmotes[outlet1] and id_dashmotes[outlet2]:
      id_dashmotes[outlet1] = id_dashmotes[outlet2]


uniques_indexes = np.where(id_dashmotes == float('nan'))

for index in uniques_indexes:
  id_dashmotes[index] = counter
  counter+=1


