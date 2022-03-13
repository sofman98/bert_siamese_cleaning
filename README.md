# dashmote
dashmote assignment

## Data Preprocessing
First split the dataset into a test-set and a train-set. This command also generates the feature similarity dataset (feature_set1, feature_set2, similarity) for both test and train data. all 4 .csv files will be stored inside the ```datasets/``` directory.
```
$python train_test_split.py
```

## Training
You can run the following command to do a grid search to look for the best combinations of hyper-parameters of the model. Feel free to open the file and change the hyper-parameters' ranges!
```
$python grid_search.py
```
The grid search results will be stored inside the ```results/grid_search_results``` directory and your last trained model will be in ```results/models/last_trained_model.h5```. 

### Interesting note:
You can modify 3 hyper-parameters, the number of dense hidden layers (range_num_dense_layers), the optimizer (range_optimizer), and the embedding size (range_embedding_size) which is the number of nodes of a sub-model's last layer. These hyper-parameters are meant for the sub-models and not directly the siamese model. The sub-models always have a decreasing number of nodes and each layer has half the nodes of its preceding layer. For example, with 3 dense layers and an embedding size of 8. the structure of the sub-models would be: [32 -> 16 -> 8]. Keep in mind that there is a Batch Normalization layer between every dense layer.

As the model is very big, if you set ```num_dense_layers``` and ```embedding_size``` to 0, you can save the outputs layer's weights only rather than the entire model, the weights will be saved inside ```results/outputs_layers/last_trained_model.npy```. If you did not, you will have to load your entire model when testing.

## Testing
Test the model by running the following command. By default, it loads the last layer's weights from ```results/outputs_layers/model_nn10.npy```. Please change ```path_to_outputs_layer``` to use your own weights, or set  ```from_outputs_layer=False``` if you want to load your entire model.
```
$python test.py
```
Running the command also saves the predictions in ```results/predictions/last_prediction.csv```.

## Generating unique IDs
After running the previous command and saving the predictions, you can use them to detect duplicates in the database and give them a same ```predicted_id_dashmote```. the resulting data will be stored in ```results/unique_ids/predicted_id_dashmote.csv```.
```
$python generate_unique_ids.py
```