from data_processing.file_management import save_csv
import pandas as pd
import numpy as np

from data_processing.file_management import load_dataset_csv, save_csv

path_to_predction = 'results/last_prediction.csv'
path_to_raw_test_data = 'datasets/test.csv'
save_resulting_test_data_to = 'results/predicted_id_dashmote.csv'
similarity_threshold = 0.5

if __name__ == "__main__":

  prediction = load_dataset_csv(path_to_predction).to_numpy()
  test = load_dataset_csv(path_to_raw_test_data)
  
  # we select the pairs whose predicted similarity exceeds our defined threshold
  duplicates_indexes = np.where(prediction[:,2] > similarity_threshold)
  prediction = prediction[duplicates_indexes]

  # we sort the predictions in a descending order
  # to give priority to the most probable duplicates
  prediction = prediction[(-prediction[:, 2]).argsort()]

  # we get the pairs
  duplicates = prediction[:, :2].astype(int)

  counter = 0
  id_dashmotes = np.full((test.shape[0], 1), -1)
  for (outlet1, outlet2) in duplicates:
    # if none of the outlets was attributed an id_dashmote
    if id_dashmotes[outlet1] < 0 and id_dashmotes[outlet2] < 0:
      id_dashmotes[outlet1] = counter
      id_dashmotes[outlet2] = counter
      counter+=1

    # if only one of them was attributed one id_dashmote
    elif id_dashmotes[outlet1] >= 0 and id_dashmotes[outlet2] < 0:
      id_dashmotes[outlet2] = id_dashmotes[outlet1]

    elif id_dashmotes[outlet1] < 0 and id_dashmotes[outlet2] >= 0:
      id_dashmotes[outlet1] = id_dashmotes[outlet2]

  # we give the remaining outlets a unique id_dashmote
  uniques_indexes = np.where(id_dashmotes < 0)[0]
  for index in uniques_indexes:
    id_dashmotes[index] = counter
    counter+=1

  id_dashmotes = pd.DataFrame(id_dashmotes, columns=['predicted_id_dashmote'], dtype=int)
  modified_test_data = pd.concat([test, id_dashmotes], axis=1)
  save_csv(modified_test_data, save_resulting_test_data_to)
  print(f'results saved in {save_resulting_test_data_to}')
