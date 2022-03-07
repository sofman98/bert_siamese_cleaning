# dashmote
dashmote assignment

## Data Processing
First split the dataset into a test-set and a train-set. This command also generates the feature similarity dataset (feature_set1, feature_set2, similarity) for both test and train data. all 4 .csv files will be stored inside the ```datasets/``` directory.
```
$python train_test_split.py
```

## Training
You can run the following command to do a grid search to look for the best combinations of hyper-parameters of the model. Feel free to open the file and change the hyper-parameters' ranges!
```
$python grid_search.py
```
The grid search results will be stored inside the ```results/``` directory and your last trained model will be in ```results/models/last_trained_model.h5```.

### Interesting note:
You can modify 3 hyper-parameters, the number of dense hidden layers (range_num_dense_layers), the optimizer (range_optimizer), and the embedding size (range_embedding_size) which is the number of nodes of a sub-model's last layer. These hyper-parameters are meant for the sub-models and not directly the siamese model. The sub-models always have a decreasing number of nodes and each layer has half the nodes of its preceding layer. For example, with 3 dense layers and an embedding size of 8. the structure of the sub-models would be: [32 -> 16 -> 8]

## Testing
Test the model by running the following command. By default, it loads ```results/models/best_model.h5```. Please change ```path_to_model``` to use your own model.
```
$python test.py
```

## Generate Embeddings
You can also generate embeddings for the test data by running the following command.
```
$python generate_embeddings.py
```
The resulted embeddings will be stored in ```datasets/test_embeddings.csv```.
