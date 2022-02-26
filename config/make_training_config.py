import json

name = 'ingredient-ResNet152-Combine-training01-batch8'

configuration = {
    # definition
    "log_filename": "./log/{}".format(name),
    "use_cuda": True,
    "epoch": 800,
    "batch_size": 8,
    "learning_rate": 0.001,
    # dataset
    "dataset_type": "IngredientDataset",
    "train_dataset": "./dataset/FruitandVegs-train.txt",
    "test_dataset": "./dataset/FruitandVegs-validation.txt",
    "num_workers": 8,
    "dataset_shuffle": True,
    "pin_memory": False,
    "crop_size": 512,
    # model
    "model_name": "ResNET152Combine",
    'last_node': 2048,
    'num_classes': 36,
    'model_checkpoint': None,
    # optimizer
    "optimizer_name": "Adam",
    "weight_decay": 0,
    "eps": 1e-08,
    "amsgrad": False,
    "betas": (0.9, 0.999),
    # tensorboard
    "tensorboard_writer_name": "./runs/{}".format(name),
    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
    "checkpoint_file_name": "{}".format(name),
    "load_checkpoint": '',
}


if __name__ == '__main__':
    filename = 'config_{}.json'.format(name)
    with open('../../config/{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')