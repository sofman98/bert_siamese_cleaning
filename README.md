# dashmote
dashmote assignment

## Data Preprocessing

### (Optional) Embedding Generation and Storing
To accelerate the training, you can generate latent vector representation of sentences then store them by running:
```
$python generate_embeddings.py
```
By default, we use the [uncased Small Bert Model](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2) with 2 layers and a hidden size of 128 to encode the outlets' name. You can change that by changing the variables ```link_to_preprocessor```, ```link_to_encoder``` and ```feature```. The embeddings will be stored in ```datasets/embeddings.npy```. Pre-computed embeddings were already saved, this step can therefore be skipped.

### Train Test Split
Split the dataset into a test-set and a train-set. This command also generates the feature similarity dataset (feature_set1, feature_set2, similarity) for both test and train data. all 4 files will be stored inside the ```datasets/``` directory. When selecting the negative instances, the default behavior is to consider all non-similar outlet pairs in a same persistent_cluster, you can change that by controlling the number of negative instances by outlets ```train_NUM_NEG```, don't forget to set ```train_max_neg = False```.
```
$python train_test_split.py
```

## Training
You can run the following command to do a grid search to look for the best combinations of hyper-parameters of the model. Feel free to open the file and change the hyper-parameters' ranges!
```
$python train.py
```
The grid search results will be stored inside the ```results/grid_search_results/``` directory and your last trained model will be in ```results/models/last_trained_model.h5```. 

### Important note:
You can modify 3 hyper-parameters, the number of dense hidden layers (range_num_dense_layers), the optimizer (range_optimizer), and the embedding size (range_embedding_size) which is the number of nodes of a sub-model's last layer. These hyper-parameters are meant for the sub-models and not directly the siamese model. The sub-models always have a decreasing number of nodes and each layer has half the nodes of its preceding layer. For example, with 3 dense layers and an embedding size of 8. the structure of the sub-models would be: [32 -> 16 -> 8]. Please, keep in mind that there is a Batch Normalization layer between every dense layer.

## Testing
Test the model by running the following command. By default, it loads the model from ```results/models/best_model.h5```. Please change ```path_to_model``` to use your own model.
```
$python test.py
```
Running the command also saves the predictions in ```results/predictions/last_prediction.csv```.

## Generating unique IDs
After running the previous command and saving the predictions, you can use them to detect duplicates in the database and give them a same ```predicted_id_dashmote```. By default, the predictions present in ```results/predictions/best_prediction.csv``` are used. The resulting data will be stored in ```results/unique_ids/predicted_id_dashmote.csv```.

```
$python generate_unique_ids.py
```