from models.transfer_learning import load_siamese_model, save_outputs_layer
import os

models_dir_path = 'results/models'
outputs_save_path = 'results/outputs_layers'

if __name__ == "__main__":
    print('saving output layers..')
    # we get all files present in models_dir_path
    models_path = os.listdir(models_dir_path)
    
    for path in models_path:
        # we load the desired model
        model = load_siamese_model(
          from_outputs_layer=False,
          path=f'{models_dir_path}/{path}'
        )
        
        # we get the file's name
        outputs_file_name = os.path.splitext(path)[0]
        # we save its last layer in a file
        save_outputs_layer(model, f'{outputs_save_path}/{outputs_file_name}')

print(f'output layers saved in {outputs_save_path}')