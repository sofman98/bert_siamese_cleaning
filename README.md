# Data Cleaning with Bert Siamese Network
## The Task
"One of our clients would like to have an overview of the Restaurants, Bars, Cafes and other F&B outlets present in the USA. To provide this service, we have obtained a collection of outlets from various food delivery platforms, such as Grubhub and Ubereats.

Unfortunately, many outlets are present on multiple platforms or have multiple listings on the same platform and are therefore duplicated in our dataset. Luckily, we managed to clean the dataset by hand by assigning each group of outlets that belong together a unique identifier.

In the future we would like to automate this process of matching outlets together. Therefore, the objective of this task is to come up with an automated solution to this problem. Please solve this challenge with Python. You can use whichever libraries you feel are appropriate for this challenge.

The dataset consists of 3,575 rows and 8 columns, where each row represents an individual outlet and each column represents a property/ feature.. Each outlet is described by its name, its address, its phone number and its geographical location. Furthermore, each outlet has a ```persistent_cluster``` and an ```id_dashmote```. The ```persistent_cluster``` specifies the rough geographical neighbourhood in which the outlet was found. The ```id_dashmote``` is a unique identifier, identifying a single cluster of associated outlets."

## The Solution
### A few important points:

  1. Some people manually did the work and created a unique ID column called ```id_dashmote``` which has the same value for duplicate outlets.
  2. In the database there is a column called ```persistent_cluster```, which is a rough geographical neighborhood of the outlet, it’s relevant because duplicates are found in a same cluster, which means that we only have to look for duplicates per cluster rather than in the entire database.

I needed to convert the database into the format (outlet1, outlet2, duplicate) with the column « duplicate » being a boolean value (0 or 1) representing whether or not the two outlets are duplicates.

The task was straightforward for positive instances in which duplicate==1, but I also needed to add some negative instances, and as there can be many more negative instances than positive ones, the dataset would be immensely imbalanced, so what I did is that I regulated the number of negative instances with a parameter I called NUM_NEG representing the number of negative instances per outlet. 

It’s important to note that not all data is equal when training this model, it’s better to train it on « hard cases ». For example, if we were to consider geographical locations as features, it would be preferable to select negative instances where two outlets are very close to each other (and still are different outlets).

Surprisingly, the geographical location inside a geographical cluster was not relevant when deciding whether two outlets are duplicates. I used the name of the restaurant (hence the BERT part), and although there were still some ways to select more relevant negative data (very similar names, different outlets) I just randomly sampled the negative instances, for now.

BERT was used as a quick solution for the problem as I needed to rapidly find a pretrained text processing model, but it is true that there could have been a more suitable model.

The testing was done on « real-life cases ». Meaning that when considering a new outlet, we would need to compare it with all of the existing outlets within a same geographical cluster. So what I did is that I dedicated a subset of the original database for testing, then I calculated all possible pairwise combinations of outlets within a same cluster. I tested my model on that (very imbalanced) data and calculated the precision, recall and confusion matrix.

Using the trained model to detect duplicates is a straightforward task. We can generate a ```predicted_id_dashmote``` column which has the same value for duplicate outlets. We first calculate all pairwise combinations of outlets per cluster, predict the probability of them being duplicates, giving a value between 0 and 1. We first only consider the pairs predicted as duplicates, say with a probability over 0.5, then we sort them according to the predicted probability, giving the higher priority to the pairs that are more likely to be duplicates. Finally we give a same unique_id to pairs that were predicted to be duplicates. We also give the outlets predicted to appear only once (non-duplicates) a unique id.

## Data Preprocessing

### (Optional) Embedding Generation and Saving
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
The grid search results will be saved inside the ```results/grid_search_results/``` directory and the model with the lowest loss value will be in saved ```results/models/last_trained_model.h5```. 

### Important note:
You can modify 3 hyper-parameters, the number of dense hidden layers (```range_num_dense_layers```), the optimizer (```range_optimizer```), and the embedding size (```range_embedding_size```) which is the number of nodes of a sub-model's last layer. These hyper-parameters are meant for the sub-models and not directly the siamese model. The sub-models always have a decreasing number of nodes and each layer has half the nodes of its preceding layer. For example, with 3 dense layers and an embedding size of 8. the structure of the sub-models would be: [32 -> 16 -> 8]. Please, keep in mind that there is a Batch Normalization layer between every dense layer.

## Testing
Test the model by running the following command. By default, it loads the model from ```results/models/best_model.h5```. Please change ```path_to_model``` to use your own model.
```
$python test.py
```
Running the command also saves the predictions in ```results/predictions/last_prediction.csv```.

## Generating unique IDs
After running the previous command and saving the predictions, you can use them to detect duplicates in the database and give them a same ```predicted_id_dashmote```. By default, the predictions present in ```results/predictions/best_prediction.csv``` are used. The resulting data will be stored in ```results/unique_ids/predicted_id_dashmote.parquet.gzip```.

```
$python generate_unique_ids.py
```
