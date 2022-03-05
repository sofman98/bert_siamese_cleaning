from data_processing.generate_similarity_dataset import generate_pair_similarity_dataset, generate_feature_similarity_dataset
from data_processing.file_management import load_dataset_csv, save_csv
import time

t1 = time.time()
test_data = load_dataset_csv('datasets/train.csv')

sim = generate_feature_similarity_dataset(
    test_data,
    features= ['lat', 'lon'],
    NUM_NEG=-1,
)

t2 = time.time()
print('time:', t2-t1)
print(sim)