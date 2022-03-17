from data_processing.file_management import load_parquet, load_csv, save_parquet, create_folder
import pandas as pd
import numpy as np

path_to_prediction = 'results/predictions/best_prediction.csv'
path_to_raw_test_data = 'datasets/test.parquet.gzip'
save_resulting_test_data_to = 'results/unique_ids/predicted_id_dashmote.parquet.gzip'
similarity_threshold = 0.5

if __name__ == "__main__":

  prediction = load_csv(path_to_prediction).to_numpy()
  test = load_parquet(path_to_raw_test_data)
  
  # we select the pairs whose predicted similarity exceeds our defined threshold
  duplicates_indexes = np.where(prediction[:,-1] > similarity_threshold)
  duplicates = prediction[duplicates_indexes]

  # we sort the predictions in a descending order
  # to give priority to the most probable duplicates
  duplicates = duplicates[(-duplicates[:, -1]).argsort()]

  # we get the pairs
  duplicate_pairs = duplicates[:, :-1].astype(int)

  counter = 0
  id_dashmotes = np.full((test.shape[0], 1), -1)
  for (outlet1, outlet2) in duplicate_pairs:
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
  create_folder(save_resulting_test_data_to)
  save_parquet(modified_test_data, save_resulting_test_data_to)
  print(f'results saved in {save_resulting_test_data_to}')
  #display
  print(modified_test_data)
