# dashmote
dashmote assignment

## Data Processing
First split the dataset into a test-set and a train-set. This command also generates the feature similarity dataset (feature_set1, feature_set2, similarity) for both test and train data. all 4 .csv files will be stored inside the datasets/ directory.
```
$ python train_test_split.py
```

## Training
You can run the following command to do a grid search to look for the best combinations of hyper-parameters of the model. Feel free to open the file and change the hyper-parameters' ranges!
```
$ python grid_search.py
```
The grid search results will be stored inside the results/ directory and your last trained model will be in results/models/last_trained_model.h5.

## Testing
Test the model by running the following command. By default, it loads results/models/best_model.h5. Please change the path_to_model variable to use your own model.
```
$ python test.py
```

## Generate Embeddings
You can also generate embeddings for the test data by running the following command.
```
$ python generate_embeddings.py
```
The resulted embeddings will be stored in datasets/test_embeddings.csv.
